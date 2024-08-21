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
from replay_buffer_ms2 import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
from ms2_utils import make_ms2_agent, convert_obs

torch.backends.cudnn.benchmark = True

@hydra.main(config_path="cfgs", config_name="config_maniskill2")
def main(cfg):

    env = gym.make(cfg.task_name, obs_mode=cfg.obs_mode, control_mode=cfg.control_mode, render_mode=cfg.render_mode)
    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    
    rgb_obs_shape = (2, 3, 84, 84)
    
    work_dir = Path.cwd()
    
    # create replay buffer
    data_specs = (
        specs.Array((2, 3, 84, 84), np.uint8, "rgb_obs"),
        specs.Array((9,), np.float32, "qpos"),
        specs.Array((8,), np.float32, "action"),
        specs.Array((1,), np.float32, "reward"),
        specs.Array((1,), np.float32, "discount"),
        specs.Array((1,), np.float32, "demo"),
    )
    
    replay_storage = ReplayBufferStorage(
        data_specs, work_dir / "buffer", True
    )
    
    replay_loader = make_replay_loader(
        work_dir / "buffer",
        cfg.replay_buffer_size,
        cfg.batch_size,
        cfg.replay_buffer_num_workers,
        cfg.save_snapshot,
        cfg.nstep,
        cfg.discount,
        cfg.do_always_bootstrap,
        cfg.frame_stack,
    )

    agent = make_ms2_agent(
                (2, 3*cfg.frame_stack, 84, 84),
                [9*cfg.frame_stack],
                [8],
                False,
                cfg.agent,
            )

    obs, _ = env.reset()
    terminated = False
    truncated = False
    global_step = 0
    
    stack_rgb_obs = np.zeros((2, 3*cfg.frame_stack, 84, 84), dtype=np.uint8)
    stack_qpos = np.zeros((9*cfg.frame_stack,), dtype=np.float32)
    
    rgb_obs, low_dim_obs = convert_obs(obs, cfg)
    
    for i in range(cfg.frame_stack):
    
        stack_rgb_obs = np.roll(stack_rgb_obs, shift=-3, axis=1)
        stack_rgb_obs[:, -3:] = rgb_obs
        stack_qpos = np.roll(stack_qpos, shift=-9, axis=0)
        stack_qpos[-9:] = low_dim_obs
        
    
    for i in range(200):
        # action = env.action_space.sample()
        
        # if i < 100:
        #     action = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        # else:
        #     action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        
        action = agent.act(torch.tensor(stack_rgb_obs).float(), torch.tensor(stack_qpos), global_step, False)
        
        cv2.imshow("rgb_obs", rgb_obs[0].transpose(1, 2, 0))
        cv2.waitKey(10)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        inst_samples = {
            'rgb_obs': rgb_obs,
            'qpos': low_dim_obs,
            'action': action,
            'reward': reward,
            'discount': 0.99,
            'demo': 0.0,
            'last': terminated or truncated,
        }
        
        replay_storage.add(inst_samples)
        
        # render = env.render()  # a display is required to render
        print("reward", reward)
        # print("render", render)
        # cv2.imshow("render", render)
        # cv2.waitKey(10)
        global_step += 1
        
        
        rgb_obs, low_dim_obs = convert_obs(obs, cfg)
        
        stack_rgb_obs = np.roll(stack_rgb_obs, shift=-3, axis=1)
        stack_rgb_obs[:, -3:] = rgb_obs
        stack_qpos = np.roll(stack_qpos, shift=-9, axis=0)
        stack_qpos[-9:] = low_dim_obs
        
    env.close()


if __name__ == "__main__":
    main()
