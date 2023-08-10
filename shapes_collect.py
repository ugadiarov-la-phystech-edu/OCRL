import os
import shutil

import numpy as np

from envs.shapes_envs.shapes_env import ShapesEnv

def create_dirs(dir_path):
    if len(dir_path) == 0:
        return

    if not os.path.isdir(dir_path):
        if os.path.isfile(dir_path):
            return
        os.makedirs(dir_path)
        
        
def remove_dir(dir_path):
    shutil.rmtree(dir_path, ignore_errors=True)      

def save_obs(img, path):
    np.save(path, img)
        
def collect_data():
    all_actions = []
    all_observations = []
    ep = 0
    env = ShapesEnv(image_size=96, 
                    task='nav_large',
                    use_agent=True)
    obs_folder = 'data/shapes_nav_10x10/val'
    steps_per_episode = 40
    episodes = 1000

    while ep < episodes:
        observations = []
        actions = []
        obs = env.reset()
        ep_dir = os.path.join(obs_folder, f'ep_{ep}')
        create_dirs(ep_dir)
        for step in range(steps_per_episode):
            action = env.action_space.sample()
            
            obs, _, done, _ = env.step(action)
            obs_dir = os.path.join(ep_dir, f'st_{step}.npy')
            save_obs(obs['image'], obs_dir)
            observations.append(obs_dir)
            
            if done:
                break
        

        if len(observations) == steps_per_episode:
            all_actions.append(actions)
            all_observations.append(observations)
            ep += 1
            if ep % 10 == 0:
                print(f'Collected {ep} episodes')
        else:
            remove_dir(ep_dir)


if __name__ == '__main__':
    collect_data()