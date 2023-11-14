import logging
from pathlib import Path

import cv2
import gym
import hydra
import omegaconf
import stable_baselines3 as sb3
import wandb
from gym import spaces
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecVideoRecorder)
from wandb.integration.sb3 import WandbCallback

import envs
import sb3s
from utils.tools import *

log = logging.getLogger(__name__)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, width: int = 64, height: int = 64):
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, env.observation_space.shape[2]),
            dtype=env.observation_space.dtype
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        return frame


class FailOnTimelimitWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        if done and 'is_success' not in info:
            info['is_success'] = False

        return observation, reward, done, info


@hydra.main(config_path="configs/", config_name="train_sb3")
def main(config):
    log_name = get_log_prefix(config)
    log_name += (
        f"-{config.sb3.name}-{config.sb3_acnet.name}-"
        f"{config.env.name}{config.env.mode}mode{config.env.rew_type}rewardtype-"
        f"Seed{config.seed}"
    )
    tags = config.tags.split(",") + config.env.tags.split(",") + [f"RandomSeed{config.seed}"]
    init_wandb(
        config,
        "TrainSB3-" + log_name,
        tags=tags,
        sync_tensorboard=True,
        monitor_gym=True,
    )

    if config.num_envs == 1:
        def make_env(seed=0):
            if config.ocr.name == "GT":
                config.env.render_mode = "state"
            if config.env.name.startswith('Navigation') or config.env.name.startswith('Pushing'):
                env = gym.make(config.env.name)
                env = WarpFrame(env, width=config.env.obs_size, height=config.env.obs_size)
                env = FailOnTimelimitWrapper(env)
                env.seed(seed)
                env.action_space.seed(seed)
            else:
                env = getattr(envs, config.env.env)(config.env, seed)
            env = Monitor(env)  # record stats such as returns
            return env
        env = DummyVecEnv([make_env])
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
                    env = WarpFrame(env, width=config.env.obs_size, height=config.env.obs_size)
                    env = FailOnTimelimitWrapper(env)
                    env.seed(seed + rank)
                    env.action_space.seed(seed + rank)
                else:
                    env = getattr(envs, config.env.env)(config.env, seed + rank)
                env = Monitor(env)  # record stats such as returns
                return env
            set_random_seed(seed)
            return _init
        env = SubprocVecEnv(
            [make_env(i, seed=config.seed) for i in range(config.num_envs)],
            start_method="fork",
        )
    # env = VecVideoRecorder(
    #     env,
    #     f"{wandb.run.dir}/videos/",
    #     record_video_trigger=lambda x: x % config.video.interval == 0,
    #     video_length=config.video.length,
    # )
    if config.ocr.name == "GT":
        config.env.render_mode = "state"
    if config.env.name.startswith('Navigation') or config.env.name.startswith('Pushing'):
        eval_env = gym.make(config.env.name)
        eval_env = WarpFrame(eval_env, width=config.env.obs_size, height=config.env.obs_size)
        eval_env = FailOnTimelimitWrapper(eval_env)
        eval_env.seed(config.seed + config.num_envs)
        eval_env.action_space.seed(config.seed + config.num_envs)
    else:
        eval_env = getattr(envs, config.env.env)(
            config.env, seed=config.seed + config.num_envs
        )
    eval_env = Monitor(eval_env)  # record stats such as returns
    model_kwargs = {
        "verbose": 1,
        "tensorboard_log": f"{wandb.run.dir}/tb_logs/",
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
    model.learn(
        total_timesteps=config.max_steps,
        log_interval=config.log_interval,
        callback=[
            WandbCallback(
                gradient_save_freq=config.wandb.log_gradient_freq,
                verbose=2,
            ),
            EvalCallback(
                eval_env,
                eval_freq=config.eval.freq,
                n_eval_episodes=config.eval.n_episodes,
                best_model_save_path=f"{wandb.run.dir}/models/",
                log_path=f"{wandb.run.dir}/eval_logs/",
                deterministic=False,
            ),
        ],
    )
    # wandb finish
    wandb.finish()


if __name__ == "__main__":
    main()
