import cv2
import gym
import hydra
import stable_baselines3 as sb3
from gym import spaces
from gym.wrappers import TimeLimit
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv)

import envs
import sb3s
from envs.maniskill3 import ManiSkillEnv
from envs.robosuite import RobosuiteEnv
from utils.tools import *

log = logging.getLogger(__name__)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, width: int = 64, height: int = 64, interpolation='cubic'):
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.interpolation = {'cubic': cv2.INTER_CUBIC, 'linear': cv2.INTER_LINEAR}[interpolation]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, env.observation_space.shape[2]),
            dtype=env.observation_space.dtype
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.resize(frame, (self.width, self.height), interpolation=self.interpolation)
        return frame


class FailOnTimelimitWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        if done and 'is_success' not in info:
            info['is_success'] = False

        return observation, reward, done, info


def make_robosuite_lift(config_env, seed=None):
    if seed is None:
        seed = config_env.seed

    env = RobosuiteEnv(config_env.name, config_env.horizon, config_env.initialization_noise_magnitude,
                       config_env.use_random_object_position, config_env.raw_observation, config_env.obs_size, seed)
    return env


def make_maniskill(config_env, seed=None):
    if seed is None:
        seed = config_env.seed

    env = ManiSkillEnv(config_env.name, config_env.obs_size, seed)
    env = TimeLimit(env, max_episode_steps=config_env.episode_length)
    return env


def create_env(config, num_envs, seed):
    if num_envs == 1:
        def make_env(seed=0):
            if config.ocr.name == "GT":
                config.env.render_mode = "state"
            if config.env.name.startswith('Navigation') or config.env.name.startswith('Pushing'):
                env = gym.make(config.env.name)
                env = WarpFrame(env, width=config.env.obs_size, height=config.env.obs_size, interpolation=config.env.interpolation)
                env = FailOnTimelimitWrapper(env)
                env.seed(seed)
                env.action_space.seed(seed)
            elif config.env.name == 'Lift':
                env = make_robosuite_lift(config.env)
            elif config.env.name == 'PushCube-v1':
                env = make_maniskill(config.env, seed=seed)
            else:
                env = getattr(envs, config.env.env)(config.env, seed)
            env = Monitor(env, info_keywords=("is_success",))  # record stats such as returns
            return env
        env = DummyVecEnv([make_env(seed)])
    else:
        def make_env(rank, seed=0):
            """
            Utility function for multiprocessed env.
                :param seed: (int) the inital seed for RNG
                :param rank: (int) index of the subprocess
            """
            def _init():
                if config.ocr.name == "GT":
                    config.env.render_mode = "state"
                if config.env.name.startswith('Navigation') or config.env.name.startswith('Pushing'):
                    env = gym.make(config.env.name)
                    env = WarpFrame(env, width=config.env.obs_size, height=config.env.obs_size, interpolation=config.env.interpolation)
                    env = FailOnTimelimitWrapper(env)
                    env.seed(seed + rank)
                    env.action_space.seed(seed + rank)
                elif config.env.name == 'Lift':
                    env = make_robosuite_lift(config.env, seed=seed + rank)
                elif config.env.name == 'PushCube-v1':
                    env = make_maniskill(config.env, seed=seed + rank)
                else:
                    env = getattr(envs, config.env.env)(config.env, seed + rank)
                env = Monitor(env, info_keywords=("is_success",))  # record stats such as returns
                return env
            return _init
        env = SubprocVecEnv(
            [make_env(i, seed=seed) for i in range(num_envs)],
            start_method="forkserver",
        )

    return env


@hydra.main(config_path="configs/", config_name="train_sb3")
def main(config):
    torch.set_float32_matmul_precision('medium')
    set_random_seed(config.seed)

    log_name = get_log_prefix(config)
    log_name += (
        f"-{config.sb3.name}-{config.sb3_acnet.name}-"
        f"{config.env.name}{config.env.mode}mode{config.env.rew_type}rewardtype-"
        f"Seed{config.seed}"
    )
    log_path = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
    env = create_env(config, num_envs=config.num_envs, seed=config.seed)
    eval_env = create_env(config, num_envs=config.eval.num_envs, seed=config.seed + config.num_envs)
    model_kwargs = {
        "verbose": 1,
        "device": config.device,
        "policy_kwargs": dict(
            features_extractor_class=sb3s.OCRExtractor,
            features_extractor_kwargs=dict(config=config),
        ),
    }
    if hasattr(config.sb3, 'algo_kwargs'):
        model_kwargs = dict(model_kwargs, **config.sb3.algo_kwargs)
    if 'n_steps' in model_kwargs:
        model_kwargs['n_steps'] = model_kwargs['n_steps'] // config.num_envs
    if hasattr(config.sb3, 'orig') and config.sb3.orig:
        policy = 'CnnPolicy'
        del model_kwargs['policy_kwargs']
    elif config.sb3_acnet.name == 'GNN':
        policy = sb3s.GNNActorCriticPolicy
        model_kwargs['policy_kwargs']['config'] = config
    else:
        policy = sb3s.CustomActorCriticPolicy
        model_kwargs['policy_kwargs']['config'] = config
    model = getattr(sb3, config.sb3.name)(
        policy,
        env,
        **model_kwargs,
    )
    freq = config.eval.freq // config.num_envs
    logger = configure(log_path, ['stdout', 'json'])
    model.set_logger(logger)
    model.learn(
        total_timesteps=config.max_steps,
        log_interval=config.log_interval,
        callback=[
            EvalCallback(
                eval_env,
                eval_freq=freq,
                n_eval_episodes=config.eval.n_episodes,
                best_model_save_path=f"{log_path}/models/",
                log_path=f"{log_path}/eval_logs/",
                deterministic=False,
            ),
        ],
    )

if __name__ == "__main__":
    main()
