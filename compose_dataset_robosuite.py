import argparse

import cv2
import h5py
import numpy as np
import robosuite
import tqdm

from robosuite import load_controller_config
from robosuite.utils.placement_samplers import UniformRandomSampler


CAMERA_NAME = 'frontview'


def wrap(obs, size=96):
    frame = obs[f'{CAMERA_NAME}_image']
    frame = np.flipud(frame)
    frame = frame[18:202, 36:220]
    frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_CUBIC)
    return frame


def make_env():
    controller_config = load_controller_config(default_controller="OSC_POSITION")
    placement_initializer = UniformRandomSampler(
        name="ObjectSampler",
        x_range=[-0.25, 0.25],
        y_range=[-0.25, 0.25],
        rotation=None,
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=np.array((0, 0, 0.8)),
        z_offset=0.01,
    )
    env = robosuite.make(
        "Lift",
        robots=["Panda"],  # load a Sawyer robot and a Panda robot
        gripper_types="default",  # use default grippers per robot arm
        controller_configs=controller_config,  # each arm is controlled using OSC
        env_configuration="default",  # (two-arm envs only) arms face each other
        use_camera_obs=True,
        use_object_obs=False,
        reward_shaping=True,
        has_renderer=False,  # on-screen rendering
        has_offscreen_renderer=True,
        control_freq=20,  # 20 hz control for applied actions
        horizon=9,  # each episode terminates after 200 steps
        camera_names="frontview",
        placement_initializer=placement_initializer,
        initialization_noise={'magnitude': 0.5, 'type': 'uniform'},
        camera_heights=256,
        camera_widths=256,
    )

    return env


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
    parser.add_argument('--train_size', type=int, default=100000)
    parser.add_argument('--val_size', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--image_size', type=int, default=96)
    args = parser.parse_args()
    image_size = args.image_size

    env = make_env()
    low, high = env.action_spec

    with h5py.File(args.dataset, 'w') as hf:
        train_group = hf.create_group('TrainingSet')
        train_dataset = train_group.create_dataset('obss', (args.train_size, image_size, image_size, 3), dtype=np.uint8)

        val_group = hf.create_group('ValidationSet')
        val_dataset = val_group.create_dataset('obss', (args.val_size, image_size, image_size, 3), dtype=np.uint8)

        n_observations_total = 0
        tqdm_bar = tqdm.tqdm(total=args.train_size + args.val_size, smoothing=0)

        while not collected:
            done = False
            observation = wrap(env.reset())
            n_observations_total += 1
            collected = collect(observation, n_observations_total, train_dataset, args.train_size, val_dataset, args.val_size, tqdm_bar)

            while not done:
                action = np.random.uniform(low, high)
                observation, _, done, _ = env.step(action)
                observation = wrap(observation)
                n_observations_total += 1
                collected = collect(observation, n_observations_total, train_dataset, args.train_size, val_dataset, args.val_size, tqdm_bar)
                if collected:
                    break

            if collected:
                break

        tqdm_bar.close()
