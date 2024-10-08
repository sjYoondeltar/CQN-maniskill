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

from dm_env import specs
from pathlib import Path

torch.backends.cudnn.benchmark = True

def make_ms2_agent(rgb_obs_shape, low_dim_obs_shape, action_shape, use_logger, cfg):
    cfg.rgb_obs_shape = rgb_obs_shape
    cfg.low_dim_obs_shape = low_dim_obs_shape
    cfg.action_shape = action_shape
    cfg.use_logger = use_logger
    return hydra.utils.instantiate(cfg)

@hydra.main(config_path="cfgs", config_name="config_maniskill2")
def main(cfg):

    env = gym.make(cfg.task_name, obs_mode=cfg.obs_mode, control_mode=cfg.control_mode, render_mode=cfg.render_mode)
    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    
    rgb_obs_shape = (2, 3, 128, 128)
    
    # create replay buffer
    data_specs = (
        specs.Array((128, 128, 3), np.uint8, "rgb_obs"),
        specs.Array((9,), np.float32, "qpos"),
        specs.Array((8,), np.float32, "action"),
        specs.Array((1,), np.float32, "reward"),
        specs.Array((1,), np.float32, "discount"),
        specs.Array((1,), np.float32, "demo"),
    )
    
    work_dir = Path.cwd()

    agent = make_ms2_agent(
                rgb_obs_shape,
                [9],
                [8],
                False,
                cfg.agent,
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
    

if __name__ == "__main__":
    main()
