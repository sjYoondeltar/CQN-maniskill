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
    # rgb -> (B, V, C, H, W) torch float tensor, V is both base_camera and hand_camera
    # resize to 84x84
    rgb_obs_list = []
    for camera in cfg.camera_keys:
        rgb_obs_list.append(cv2.resize(obs['image'][camera]['rgb'], (cfg.camera_shape[0], cfg.camera_shape[1])).transpose(2, 0, 1).reshape(1, 3, cfg.camera_shape[0], cfg.camera_shape[1]))
    rgb_obs = np.concatenate(rgb_obs_list, axis=0)
    
    low_dim_obs = obs['agent'][cfg.state_keys[0]]
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
