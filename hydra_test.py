import logging
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import gymnasium as gym
import mani_skill2.envs
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True

def make_agent(rgb_obs_spec, low_dim_obs_spec, action_spec, use_logger, cfg):
    cfg.rgb_obs_shape = rgb_obs_spec.shape
    cfg.low_dim_obs_shape = low_dim_obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.use_logger = use_logger
    return hydra.utils.instantiate(cfg)

@hydra.main(config_path="cfgs", config_name="config_maniskill2")
def main(cfg):

    env = gym.make("PickCube-v0", obs_mode="rgbd", control_mode="pd_joint_delta_pos", render_mode="cameras")
    print("Observation space", env.observation_space)
    print("Action space", env.action_space)

    agent = make_agent(
                self.train_env.rgb_observation_spec(),
                self.train_env.low_dim_observation_spec(),
                self.train_env.action_spec(),
                False,
                cfg.agent,
            )

if __name__ == "__main__":
    main()
