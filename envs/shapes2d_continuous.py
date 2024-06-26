import collections

import gym
import numpy as np
from gym.utils import seeding
from gym import spaces

import envs
from envs.collision_system import CollisionSystem
from envs.particle import Particle


class Shapes2dContinuous(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    STEP_REWARD = -0.01
    OUT_OF_FIELD_REWARD = -0.1
    COLLISION_REWARD = -0.1
    DEATH_REWARD = -1
    HIT_GOAL_REWARD = 1
    DESTROY_GOAL_REWARD = -1

    def __init__(self, n_active_objects=1, n_passive_objects=3, n_goal_objects=1, seed=None, render_mode="rgb_array",
                 do_reward_active_objects=False, do_push_out_active_objects=False, do_push_out_passive_objects=False,
                 velocity=0.02, friction=0.0003, simulation_time_limit=10000, visualize=False, frame_rate=25,
                 object_size=10, field_size=70):
        assert render_mode in self.metadata["render.modes"], "Invalid render mode"
        self.render_mode = render_mode

        self.n_active_objects = n_active_objects
        self.n_passive_objects = n_passive_objects
        self.n_goal_objects = n_goal_objects
        self.object_kinds = collections.defaultdict(list)
        for i in range(self.n_active_objects + self.n_passive_objects + self.n_goal_objects):
            if i < self.n_active_objects:
                self.object_kinds['active'].append(i)
            elif i < self.n_active_objects + self.n_passive_objects:
                self.object_kinds['passive'].append(i)
            elif i < self.n_active_objects + self.n_passive_objects + self.n_goal_objects:
                self.object_kinds['goal'].append(i)
            else:
                assert False, 'Cannot happen!'

        self.successful_interactions_counter = None
        self.do_reward_active_objects = do_reward_active_objects
        self.do_push_out_active_objects = do_push_out_active_objects
        self.do_push_out_passive_objects = do_push_out_passive_objects

        self.object_size = object_size
        self.field_size = field_size
        self.velocity = velocity
        self.friction = friction
        self.action_space = spaces.Box(
            low=np.array([0, 0] * self.n_active_objects, dtype=np.float32),
            high=np.array([self.velocity, 2 * np.pi] * self.n_active_objects, dtype=np.float32),
            seed=seed
        )
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.field_size, self.field_size, 3),
                                            dtype=np.uint8, seed=seed)
        self.collision_system = CollisionSystem(
            n_particles=self.n_active_objects + self.n_passive_objects + self.n_goal_objects,
            object_size=self.object_size, field_size=self.field_size, simulation_limit=simulation_time_limit,
            active_disappear_on_hit_goal=self.do_reward_active_objects,
            active_disappear_on_hit_wall=self.do_push_out_active_objects,
            passive_disappear_on_hit_wall=self.do_push_out_passive_objects, visualize=visualize, frame_rate=frame_rate)

        self.np_random = None
        self.steps_taken = None
        self.objects = None
        self.current_observation = None
        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode=None):
        return self.current_observation

    def add_particle_non_overlapping(self, kind, off_wall_placement):
        index = len(self.objects)
        while True:
            particle = Particle.create(index, self.np_random, kind, self.object_size / self.field_size,
                                       off_wall_placement, velocity=0, friction=self.friction)
            does_intersect = False
            for existing_particle in self.objects:
                if particle.does_intersect_with_particle(existing_particle):
                    does_intersect = True
                    break

            if not does_intersect:
                self.objects.append(particle)
                break

    def reset(self, seed=None, options=None):
        self.objects = []
        for _ in range(self.n_active_objects):
            self.add_particle_non_overlapping('active', off_wall_placement=False)

        for _ in range(self.n_passive_objects):
            self.add_particle_non_overlapping('passive', off_wall_placement=False)

        for _ in range(self.n_goal_objects):
            self.add_particle_non_overlapping('goal', off_wall_placement=False)

        assert len(self.objects) == self.n_active_objects + self.n_passive_objects + self.n_goal_objects, \
            (f'len(self.objects)={len(self.objects)}, '
             + f'expected: {self.n_active_objects + self.n_passive_objects + self.n_goal_objects}')

        self.steps_taken = 0
        self.successful_interactions_counter = self.n_passive_objects
        if self.do_reward_active_objects:
            self.successful_interactions_counter += self.n_active_objects

        self.collision_system.set_particles(self.objects)
        self.current_observation = self.collision_system.get_observation()
        info = {'is_success': False}

        return self.current_observation

    def step(self, action):
        action = action.reshape(self.n_active_objects, 2)
        for obj, (velocity, angle) in zip(self.objects[:self.n_active_objects], action):
            if obj is not None:
                obj.set_velocity(velocity=velocity, angle=angle)

        self.collision_system.reset()
        interaction_dict = self.collision_system.simulate()
        self.successful_interactions_counter -= interaction_dict[self.collision_system.PASSIVE_GOAL]
        if self.do_reward_active_objects:
            self.successful_interactions_counter -= interaction_dict[self.collision_system.ACTIVE_GOAL]
            terminated = all(self.objects[i] is None for i in self.object_kinds['active'])
        else:
            terminated = all(self.objects[i] is None for i in self.object_kinds['passive'])

        info = {'is_success': self.successful_interactions_counter == 0}
        reward = self.STEP_REWARD
        reward += self.HIT_GOAL_REWARD * interaction_dict[self.collision_system.PASSIVE_GOAL]
        reward += self.COLLISION_REWARD * interaction_dict[self.collision_system.ACTIVE_ACTIVE]
        if not self.do_reward_active_objects:
            reward += self.COLLISION_REWARD * interaction_dict[self.collision_system.ACTIVE_GOAL]
        else:
            reward += self.HIT_GOAL_REWARD * interaction_dict[self.collision_system.ACTIVE_GOAL]

        if self.do_push_out_active_objects:
            reward += self.DEATH_REWARD * interaction_dict[self.collision_system.ACTIVE_WALL]
        else:
            reward += self.OUT_OF_FIELD_REWARD * interaction_dict[self.collision_system.ACTIVE_WALL]

        if self.do_push_out_passive_objects:
            reward += self.DEATH_REWARD * interaction_dict[self.collision_system.PASSIVE_WALL]
        else:
            reward += self.OUT_OF_FIELD_REWARD * interaction_dict[self.collision_system.PASSIVE_WALL]

        return self.collision_system.get_observation(), reward, terminated, info


if __name__ == '__main__':
    # env = gym.make('Pushing7x7Continuous-v0', visualize=True, frame_rate=50)
    # env = gym.make('Pushing7x7ContinuousActiveObjects-v0', visualize=True, frame_rate=50)
    # env = gym.make('Pushing5x5ContinuousSlow-v0', visualize=True, frame_rate=50)
    # env = gym.make('Navigation7x7Continuous-v0', visualize=True, frame_rate=50)
    # env = gym.make('Navigation7x7ContinuousOpenField-v0', visualize=True, frame_rate=50)
    # env = gym.make('Navigation7x7ContinuousSlow-v0', visualize=True, frame_rate=50)
    env = gym.make('Navigation5x5ContinuousSlow-v0', visualize=True, frame_rate=50)
    obs = env.reset()
    done = False
    episode_return = 0
    while not done:
        action = env.action_space.sample()
        o, r, terminated, info = env.step(action)
        episode_return += r
        print(f'info={info}')
        print(f'reward={r} terminated={terminated}')
        done = terminated

    print(f'return={episode_return}')
