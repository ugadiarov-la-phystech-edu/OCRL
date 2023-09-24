import gym
import gym.wrappers
import numpy as np
import cv2
from envs.shapes_envs import shapes2d
from gym.envs.registration import register

def resolve_task(task, **task_kwargs):
    if task == 'nav_small':
        return gym.make('Navigation5x5-v0', **task_kwargs)
    elif task == 'random_walk':
        return gym.make('RandomWalk-v0', **task_kwargs)
    elif task == 'nav_large':
        return gym.make('Navigation10x10-v0', **task_kwargs)
    elif task == 'push_small':
        return gym.make('Pushing7x7-v0', **task_kwargs)
    elif task == 'push-no-agent_small':
        return gym.make('PushingNoAgent5x5-v0', **task_kwargs)
    else:
        ValueError(f"Task {task} is not supported")
        
class ShapesEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}
    
    def __init__(self, config, seed):
        self._config = config
        self.env = resolve_task(self._config.task, seed=seed, **self._config.task_kwargs)
        self.image_size = self._config.obs_size
        self.agent = None
        if self._config.use_agent:
            self.agent = shapes2d.AdHocNavigationAgent(self.env)
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=255, 
            shape=(self._config.obs_channels, self.image_size, self.image_size), 
            dtype=np.uint8
        ) 
        # self.observation_space = gym.spaces.MultiBinary((self._config.obs_channels, self.image_size, self.image_size))
        
    
    def _process_image(self, image):
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
            
        return image
    
    def render(self, mode=None):
        img = self.env.render(mode)
        return self._process_image(img)
        
    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        image = self.env.reset()[0][1]
        image = self._process_image(image)
        obs = image.transpose(2, 0, 1)
        return obs
    
    def step(self, action):
        if self.agent is not None:
            action = self.agent.act(0,0,0)
        state, reward, done, info = self.env.step(action)
        image = self._process_image(state[1])
        obs = image.transpose(2, 0, 1)
        return obs, reward, done, info
    