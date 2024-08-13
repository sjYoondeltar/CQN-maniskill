import hydra
import torch
import cv2
import h5py

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


# loads h5 data into memory for faster access
def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out
