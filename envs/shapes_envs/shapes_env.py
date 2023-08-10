import gym
import gym.wrappers
import numpy as np
import cv2
from envs.shapes_envs import shapes2d
from gym.envs.registration import register

def resolve_task(task, *args, **kwargs):
    if task == 'nav_small':
        return gym.make('Navigation5x5-v0', *args, **kwargs)
    elif task == 'random_walk':
        return gym.make('RandomWalk-v0', *args, **kwargs)
    elif task == 'nav_large':
        return gym.make('Navigation10x10-v0', *args, **kwargs)
    else:
        ValueError(f"Task {task} is not supported")
        
class ShapesEnv(gym.Env):
    def __init__(self, image_size=64, use_agent=False, task='nav_small', *args, **kwargs):
        self.env = resolve_task(task, *args, **kwargs)
        self.image_size = image_size
        self.agent = None
        if use_agent:
            self.agent = shapes2d.AdHocNavigationAgent(self.env)
        
        c,h,w = self.reset()['image'].shape
        self.observation_space = gym.spaces.MultiBinary((c, self.image_size, self.image_size))
        
    
    def _process_image(self, image):
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
            
        return image
        
        
    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        image = self.env.reset()[0][1]
        image = self._process_image(image)
        obs = {"image": image.transpose(2, 0, 1)}
        return obs
    
    def step(self, action):
        if self.agent is not None:
            action = self.agent.act(0,0,0)
        state, reward, done, info = self.env.step(action)
        image = self._process_image(state[1])
        obs = {"image": image.transpose(2, 0, 1)}
        return obs, reward, done, info
    