from gym.envs.registration import register

from .synthetic_envs import RandomObjsEnv, OddOneOutEnv, TargetEnv, PushEnv, MazeEnv
from .cw_envs import CwTargetEnv


register(
    'Navigation5x5-v0',
    entry_point='envs.shapes2d:Shapes2d',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 1,
        'n_static_boxes': 0,
        'static_goals': True,
        'width': 5,
        'render_scale': 10,
        'channel_wise': False,
        'channels_first': False,
        'ternary_interactions': False,
        'embodied_agent': False,
        'do_reward_push_only': False,
    },
)

register(
    'Navigation7x7-v0',
    entry_point='envs.shapes2d:Shapes2d',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 1,
        'n_static_boxes': 0,
        'static_goals': True,
        'width': 7,
        'render_scale': 10,
        'channel_wise': False,
        'channels_first': False,
        'ternary_interactions': False,
        'embodied_agent': False,
        'do_reward_push_only': False,
    },
)

register(
    'Navigation7x7Random-v0',
    entry_point='envs.shapes2d:Shapes2d',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 1,
        'n_static_boxes': 0,
        'static_goals': True,
        'width': 7,
        'render_scale': 10,
        'channel_wise': False,
        'channels_first': False,
        'ternary_interactions': False,
        'embodied_agent': False,
        'do_reward_push_only': False,
        'use_random_shapes': True
    },
)

register(
    'Navigation10x10-v0',
    entry_point='envs.shapes2d:Shapes2d',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 8,
        'n_goals': 1,
        'n_static_boxes': 0,
        'static_goals': True,
        'width': 10,
        'render_scale': 10,
        'channel_wise': False,
        'channels_first': False,
        'ternary_interactions': False,
        'embodied_agent': False,
        'do_reward_push_only': False,
    },
)

register(
    'Pushing7x7-v0',
    entry_point='envs.shapes2d:Shapes2d',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 1,
        'n_static_boxes': 0,
        'width': 7,
        'render_scale': 10,
        'channel_wise': False,
        'channels_first': False,
        'ternary_interactions': True,
        'embodied_agent': True,
        'do_reward_push_only': False,
    },
)

register(
    'PushingNoAgent5x5-v0',
    entry_point='envs.shapes2d:Shapes2d',
    max_episode_steps=100,
    kwargs={
        'observation_type': 'shapes',
        'border_walls': True,
        'n_boxes': 5,
        'n_goals': 1,
        'n_static_boxes': 0,
        'width': 5,
        'render_scale': 10,
        'channel_wise': False,
        'channels_first': False,
        'ternary_interactions': True,
        'embodied_agent': False,
        'do_reward_push_only': True,
    },
)

register(
    'Pushing7x7Continuous-v0',
    entry_point='envs.shapes2d_continuous:Shapes2dContinuous',
    max_episode_steps=100,
    kwargs={
        'n_active_objects': 1,
        'n_passive_objects': 3,
        'n_goal_objects': 1,
        'velocity': 0.02,
        'do_reward_active_objects': False,
        'do_push_out_active_objects': False,
        'do_push_out_passive_objects': False,
    },
)

register(
    'Pushing7x7ContinuousActiveObjects-v0',
    entry_point='envs.shapes2d_continuous:Shapes2dContinuous',
    max_episode_steps=100,
    kwargs={
        'n_active_objects': 1,
        'n_passive_objects': 3,
        'n_goal_objects': 1,
        'velocity': 0.02,
        'do_reward_active_objects': True,
        'do_push_out_active_objects': False,
        'do_push_out_passive_objects': False,
    },
)

register(
    'Pushing7x7ContinuousSlow-v0',
    entry_point='envs.shapes2d_continuous:Shapes2dContinuous',
    max_episode_steps=100,
    kwargs={
        'n_active_objects': 1,
        'n_passive_objects': 3,
        'n_goal_objects': 1,
        'velocity': 0.01,
        'do_reward_active_objects': False,
        'do_push_out_active_objects': False,
        'do_push_out_passive_objects': False,
    },
)

register(
    'Navigation7x7Continuous-v0',
    entry_point='envs.shapes2d_continuous:Shapes2dContinuous',
    max_episode_steps=100,
    kwargs={
        'n_active_objects': 4,
        'n_passive_objects': 0,
        'n_goal_objects': 1,
        'velocity': 0.015,
        'do_reward_active_objects': True,
        'do_push_out_active_objects': False,
        'do_push_out_passive_objects': False,
    },
)

register(
    'Navigation7x7ContinuousOpenField-v0',
    entry_point='envs.shapes2d_continuous:Shapes2dContinuous',
    max_episode_steps=100,
    kwargs={
        'n_active_objects': 4,
        'n_passive_objects': 0,
        'n_goal_objects': 1,
        'velocity': 0.015,
        'do_reward_active_objects': True,
        'do_push_out_active_objects': True,
        'do_push_out_passive_objects': False,
    },
)

register(
    'Navigation7x7ContinuousSlow-v0',
    entry_point='envs.shapes2d_continuous:Shapes2dContinuous',
    max_episode_steps=100,
    kwargs={
        'n_active_objects': 4,
        'n_passive_objects': 0,
        'n_goal_objects': 1,
        'velocity': 0.01,
        'do_reward_active_objects': True,
        'do_push_out_active_objects': False,
        'do_push_out_passive_objects': False,
    },
)
