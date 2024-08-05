import logging
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import cv2
import gymnasium as gym
import mani_skill2.envs
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True

def make_ms2_agent(rgb_obs_shape, low_dim_obs_shape, action_shape, use_logger, cfg):
    cfg.rgb_obs_shape = rgb_obs_shape
    cfg.low_dim_obs_shape = low_dim_obs_shape
    cfg.action_shape = action_shape
    cfg.use_logger = use_logger
    return hydra.utils.instantiate(cfg)

def convert_obs(obs, cfg):
    # rgb -> (B, V, C, H, W) torch float tensor, V is both base_camera and hand_camera
    # resize to 84x84
    rgb_obs_list = []
    for camera in ['base_camera', 'hand_camera']:
        rgb_obs_list.append(torch.tensor(cv2.resize(obs['image'][camera]['rgb'], (84, 84)).transpose(2, 0, 1), dtype=torch.float32)[None])
    rgb_obs = torch.cat(rgb_obs_list, dim=0)
    
    low_dim_obs = torch.tensor(obs['agent'][cfg.state_keys[0]], dtype=torch.float32)
    return rgb_obs, low_dim_obs

@hydra.main(config_path="cfgs", config_name="config_maniskill2")
def main(cfg):

    env = gym.make(cfg.task_name, obs_mode=cfg.obs_mode, control_mode=cfg.control_mode, render_mode=cfg.render_mode)
    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    
    rgb_obs_shape = (2, 3, 84, 84)

    agent = make_ms2_agent(
                rgb_obs_shape,
                [9],
                [8],
                False,
                cfg.agent,
            )

    obs, _ = env.reset()
    done = False
    global_step = 0
    for i in range(200):
        # action = env.action_space.sample()
        
        # if i < 100:
        #     action = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        # else:
        #     action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        
        rgb_obs, low_dim_obs = convert_obs(obs, cfg)
        action = agent.act(rgb_obs, low_dim_obs, global_step, True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        # render = env.render()  # a display is required to render
        print("reward", reward)
        # print("render", render)
        # cv2.imshow("render", render)
        # cv2.waitKey(10)
        global_step += 1
        
    env.close()


if __name__ == "__main__":
    main()
