from typing import Any, Dict

import gym
import numpy as np

import gymnasium
import mani_skill.envs
import torch
from PIL import Image
from mani_skill import PushCubeEnv
from mani_skill.utils.structs import Array, Pose


class PushCubeCustomEnv(PushCubeEnv):
    def __init__(self, *args, pose_reward_coef=1., place_reward_coef=1., **kwargs):
        super().__init__(*args, **kwargs)
        self._pose_reward_coef = pose_reward_coef
        self._place_reward_coef = place_reward_coef

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # We also create a pose marking where the robot should push the cube from that is easiest (pushing from behind the cube)
        tcp_push_pose = Pose.create_from_pq(
            p=self.obj.pose.p
            + torch.tensor([-self.cube_half_size - 0.005, 0, 0], device=self.device)
        )
        tcp_to_push_pose = tcp_push_pose.p - self.agent.tcp.pose.p
        tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_push_pose_dist)
        reward = self._pose_reward_coef * reaching_reward

        # compute a placement reward to encourage robot to move the cube to the center of the goal region
        # we further multiply the place_reward by a mask reached so we only add the place reward if the robot has reached the desired push pose
        # This reward design helps train RL agents faster by staging the reward out.
        reached = tcp_to_push_pose_dist < 0.01
        obj_to_goal_dist = torch.linalg.norm(
            self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += self._place_reward_coef * place_reward * reached

        # assign rewards to parallel environments that achieved success to the maximum of 3.
        reward[info["success"]] = 3
        return reward


class ManiSkillEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, name, obs_size, seed):
        self._name = name
        self._obs_size = obs_size
        self._seed = seed
        self._env = gymnasium.make(
            self._name,
            obs_mode='rgb+segmentation',
            control_mode='pd_joint_delta_pos',
            render_mode='rgb_array',
            sensor_configs=dict(width=self._obs_size, height=self._obs_size),
        ).env
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self._obs_size, self._obs_size, 3),
                                                dtype=np.uint8)
        self.action_space = gym.spaces.Box(
            low=np.full(self._env.action_space.shape, self._env.action_space.low.min()),
            high=np.full(self._env.action_space.shape, self._env.action_space.high.max()),
            dtype=self._env.action_space.dtype,
        )
        self._env.reset(seed=self._seed)
        self.last_observation = None

    def _unravel(self, step_result):
        visual_data = step_result[0]['sensor_data']['base_camera']
        rgb = visual_data['rgb'][0]
        unravel_result = [rgb]
        unravel_result += [x[0] if hasattr(x, '__len__') else x for x in step_result[1:-1]]
        info = {key: value[0] if hasattr(value, '__len__') else value for key, value in step_result[-1].items()}
        unravel_result.append(info)

        return unravel_result

    def _process_observation(self, observation):
        self.last_source_observation = observation.numpy()
        self.last_observation = np.array(
            Image.fromarray(self.last_source_observation).resize((self._obs_size, self._obs_size))
        )
        return self.last_observation.copy()

    def reset(self, *args, **kwargs):
        obs = self._unravel(self._env.reset())[0]
        return self._process_observation(obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._unravel(self._env.step(action))
        info = {k: v.item() for k, v in info.items()}
        info["success"] = int(info.get("success", 0))
        info["is_success"] = bool(info["success"])
        return self._process_observation(obs), reward.item(), terminated.item() or truncated.item(), info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        return self.last_observation.copy()
