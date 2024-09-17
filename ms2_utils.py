import hydra
import torch
import cv2
import h5py
import numpy as np

def make_ms2_agent(rgb_obs_shape, low_dim_obs_shape, action_shape, use_logger, cfg):
    cfg.rgb_obs_shape = rgb_obs_shape
    cfg.low_dim_obs_shape = low_dim_obs_shape
    cfg.action_shape = action_shape
    cfg.use_logger = use_logger
    return hydra.utils.instantiate(cfg)


def convert_obs(obs, cfg):
    rgb_obs_list = []
    for camera in cfg.observation.camera_keys:
        rgb_obs = cv2.resize(obs['image'][camera]['rgb'], tuple(cfg.observation.camera_shape))
        rgb_obs = rgb_obs.transpose(2, 0, 1).reshape(1, 3, *cfg.observation.camera_shape)
        rgb_obs_list.append(rgb_obs)
    rgb_obs = np.concatenate(rgb_obs_list, axis=0)
    
    low_dim_obs_list = []
    total_dim = 0
    for item in cfg.observation.low_dim_keys:
        key, expected_dim = item['key'], item['dim']
        if key == "goal_relative":
            value = obs["extra"]["goal_pos"] - obs["extra"]["tcp_pose"][:3]
        elif key in obs["agent"]:
            value = obs["agent"][key]
        elif key in obs["extra"]:
            value = obs["extra"][key]
        else:
            raise ValueError(f"Unknown key: {key}")
        
        if value.shape[-1] != expected_dim:
            raise ValueError(f"Dimension mismatch for {key}: expected {expected_dim}, got {value.shape[-1]}")
        
        low_dim_obs_list.append(value)
        total_dim += expected_dim
    
    low_dim_obs = np.concatenate(low_dim_obs_list, axis=0)
    assert low_dim_obs.shape[-1] == total_dim, f"Total dimension mismatch: expected {total_dim}, got {low_dim_obs.shape[-1]}"
    
    return rgb_obs, low_dim_obs

# loads h5 data into memory for faster access
def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out
