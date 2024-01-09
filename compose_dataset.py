import argparse

import cv2
import gym
import h5py
import numpy as np
import tqdm
from gym import spaces

from gym.wrappers import TimeLimit
from omegaconf import OmegaConf

from envs.cw_envs import CwTargetEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, width: int = 64, height: int = 64):
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, env.observation_space.shape[2]), dtype=env.observation_space.dtype
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        return frame


def make_env(path, seed):
    def init():
        env_config = OmegaConf.load(path)
        set_random_seed(seed)
        env = CwTargetEnv(env_config, seed)
        env.action_space.seed(seed)
        env = TimeLimit(env, env.unwrapped._max_episode_length)
        env = Monitor(env)
        return env

    return init


def make_gym(env_id, image_size, seed):
    def init():
        env = gym.make(env_id, seed=seed)
        env.action_space.seed(seed)
        env = ResizeWrapper(env, image_size, image_size)
        env = Monitor(env)
        return env

    return init


def collect(observation, n_observations_total, train_dataset, train_size, val_dataset, val_size, tqdm_bar):
    if n_observations_total <= val_size:
        val_dataset[n_observations_total - 1, ...] = observation
    elif n_observations_total <= val_size + train_size:
        train_dataset[n_observations_total - val_size - 1, ...] = observation
    else:
        return True

    tqdm_bar.update(1)
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='dataset.hdf5')
    parser.add_argument('--n_envs', type=int, default=1)
    parser.add_argument('--train_size', type=int, default=100000)
    parser.add_argument('--val_size', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env_type', choices=['gym', 'cw'], default='cw')
    parser.add_argument('--env_config', type=str)
    parser.add_argument('--env_id', type=str)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--vec_env', type=str, choices=['dummy', 'subproc'])
    args = parser.parse_args()
    if args.env_type == 'gym':
        assert hasattr(args, 'env_id'), f'env_id argument must be provided for gym environment type'
    elif args.env_type == 'cw':
        assert hasattr(args, 'env_config'), f'env_config argument must be provided for cw environment type'
    else:
        assert False, 'Cannot happen!'

    image_size = args.image_size
    if args.vec_env == 'dummy':
        vec_env_type = DummyVecEnv
    elif args.vec_env == 'subproc':
        vec_env_type = SubprocVecEnv
    else:
        assert False, 'Cannot happen!'

    if args.env_type == 'cw':
        env = vec_env_type([make_env(args.env_config, args.seed + i) for i in range(args.n_envs)])
    elif args.env_type == 'gym':
        env = vec_env_type([make_gym(args.env_id, image_size, args.seed + i) for i in range(args.n_envs)])
    else:
        assert False, 'Cannot happen!'

    action_spaces = env.get_attr('action_space')

    with h5py.File(args.dataset, 'w') as hf:
        train_group = hf.create_group('TrainingSet')
        train_dataset = train_group.create_dataset('obss', (args.train_size, image_size, image_size, 3), dtype=np.uint8)

        val_group = hf.create_group('ValidationSet')
        val_dataset = val_group.create_dataset('obss', (args.val_size, image_size, image_size, 3), dtype=np.uint8)

        n_observations_total = 0
        tqdm_bar = tqdm.tqdm(total=args.train_size + args.val_size, smoothing=0)

        observations = env.reset()
        for observation in observations:
            n_observations_total += 1
            collected = collect(observation, n_observations_total, train_dataset, args.train_size, val_dataset, args.val_size, tqdm_bar)

        while not collected:
            actions = [action_space.sample() for action_space in action_spaces]
            for observation, _, done, info in zip(*env.step(actions)):
                n_observations_total += 1
                collected = collect(observation, n_observations_total, train_dataset, args.train_size, val_dataset, args.val_size, tqdm_bar)
                if collected:
                    break

                if done:
                    observation = info["terminal_observation"]
                    n_observations_total += 1
                    collected = collect(observation, n_observations_total, train_dataset, args.train_size, val_dataset, args.val_size, tqdm_bar)
                    if collected:
                        break

        tqdm_bar.close()
