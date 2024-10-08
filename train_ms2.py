import logging
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import hydra
import numpy as np
import cv2
import torch
from dm_env import specs

import gymnasium as gym
import mani_skill2.envs
import utils
from logger import Logger
from replay_buffer_ms2 import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
from ms2_utils import make_ms2_agent, convert_obs, load_h5_data

torch.backends.cudnn.benchmark = True


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        
        if self.cfg.control_mode == "pd_ee_delta_pose":
            self.cfg.agent.action_shape = 7
        else:
            self.cfg.agent.action_shape = 8

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()
        
        self.agent = make_ms2_agent(
            (len(self.cfg.observation.camera_keys), 3*self.cfg.frame_stack, self.cfg.observation.camera_shape[0], self.cfg.observation.camera_shape[1]),
            [sum(item['dim'] for item in self.cfg.observation.low_dim_keys)*self.cfg.frame_stack],
            [self.cfg.agent.action_shape],
            False,
            self.cfg.agent,
        )
        self.timer = utils.Timer()
        self.logger = Logger(
            self.work_dir, self.cfg.use_tb, self.cfg.use_wandb, self.cfg
        )
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create envs
        if self.cfg.task_name == "PickSingleYCB-v0":
            self.train_env = gym.make(
                self.cfg.task_name,
                obs_mode=self.cfg.obs_mode,
                control_mode=self.cfg.control_mode,
                render_mode=self.cfg.render_mode,
                model_ids=self.cfg.model_ids,
            )
        else:
            self.train_env = gym.make(
                self.cfg.task_name,
                obs_mode=self.cfg.obs_mode,
                control_mode=self.cfg.control_mode,
                render_mode=self.cfg.render_mode,
                camera_cfgs={
                    "width": self.cfg.observation.camera_shape[0],
                    "height": self.cfg.observation.camera_shape[1],
                }
            )
        # create replay buffer
        data_specs = (
            specs.Array((len(self.cfg.observation.camera_keys), 3, self.cfg.observation.camera_shape[0], self.cfg.observation.camera_shape[1]), np.uint8, "rgb_obs"),
            specs.Array((sum(item['dim'] for item in self.cfg.observation.low_dim_keys),), np.float32, "qpos"),
            specs.Array((self.cfg.agent.action_shape,), np.float32, "action"),
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
            specs.Array((1,), np.float32, "demo"),
        )
        
        self.stack_rgb_obs = np.zeros((len(self.cfg.observation.camera_keys), 3*self.cfg.frame_stack, self.cfg.observation.camera_shape[0], self.cfg.observation.camera_shape[1]), dtype=np.uint8)
        self.stack_qpos = np.zeros((sum(item['dim'] for item in self.cfg.observation.low_dim_keys)*self.cfg.frame_stack,), dtype=np.float32)

        self.replay_storage = ReplayBufferStorage(
            data_specs, self.work_dir / "buffer", self.cfg.use_relabeling
        )
        self.demo_replay_storage = ReplayBufferStorage(
            data_specs,
            self.work_dir / "demo_buffer",
            self.cfg.use_relabeling,
            is_demo_buffer=True,
        )

        self.replay_loader = make_replay_loader(
            self.work_dir / "buffer",
            self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot,
            self.cfg.nstep,
            self.cfg.discount,
            self.cfg.do_always_bootstrap,
            self.cfg.frame_stack,
        )
        self.demo_replay_loader = make_replay_loader(
            self.work_dir / "demo_buffer",
            self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot,
            self.cfg.nstep,
            self.cfg.discount,
            self.cfg.do_always_bootstrap,
            self.cfg.frame_stack,
        )
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None
        )

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            replay_iter = iter(self.replay_loader)
            demo_replay_iter = iter(self.demo_replay_loader)
            self._replay_iter = utils.DemoMergedIterator(replay_iter, demo_replay_iter)
        return self._replay_iter

    def eval(self):
        """We use train env for evaluation, because it's convenient"""
        step, episode, total_reward = 0, 0, 0
        
        self.stack_rgb_obs = np.zeros((len(self.cfg.observation.camera_keys), 3*self.cfg.frame_stack, self.cfg.observation.camera_shape[0], self.cfg.observation.camera_shape[1]), dtype=np.uint8)
        self.stack_qpos = np.zeros((sum(item['dim'] for item in self.cfg.observation.low_dim_keys)*self.cfg.frame_stack,), dtype=np.float32)

        obs, _ = self.train_env.reset()
        terminated = False
        truncated = False
        
        rgb_obs, low_dim_obs = convert_obs(obs, self.cfg)
        # stack_rgb_obs, stack_low_dim_obs = self.update_frame_stack(rgb_obs, low_dim_obs, self.cfg.low_dim_obs_shape)
        for _ in range(self.cfg.frame_stack):
            stack_rgb_obs, stack_low_dim_obs = self.update_frame_stack(rgb_obs, low_dim_obs, sum(item['dim'] for item in self.cfg.observation.low_dim_keys))
        
        self.video_recorder.init(self.train_env, enabled=(episode == 0))
        while not (terminated or truncated):
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(
                    stack_rgb_obs,
                    stack_low_dim_obs,
                    self.global_step,
                    eval_mode=False,
                )
            obs, reward, terminated, truncated, info  = self.train_env.step(action)
            rgb_obs, low_dim_obs = convert_obs(obs, self.cfg)
            stack_rgb_obs, stack_low_dim_obs = self.update_frame_stack(rgb_obs, low_dim_obs, sum(item['dim'] for item in self.cfg.observation.low_dim_keys))
            self.video_recorder.record(self.train_env)
            total_reward += reward
            step += 1

        episode += 1
        self.video_recorder.save(f"{self.global_frame}.mp4")

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode_reward", total_reward / episode)
            log("episode_length", step * self.cfg.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)
            
    def update_frame_stack(self, rgb_obs, low_dim_obs, low_dim):
        self.stack_rgb_obs = np.roll(self.stack_rgb_obs, shift=-3, axis=1)
        self.stack_rgb_obs[:, -3:] = rgb_obs
        self.stack_qpos = np.roll(self.stack_qpos, shift=-low_dim, axis=0)
        self.stack_qpos[-low_dim:] = low_dim_obs
        return self.stack_rgb_obs, self.stack_qpos

    def train(self):
        # predicates
        train_until_step = utils.Until(
            self.cfg.num_train_frames, self.cfg.action_repeat
        )
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(
            self.cfg.eval_every_frames, self.cfg.action_repeat
        )

        do_eval = False

        episode_step, episode_reward = 0, 0
        
        self.stack_rgb_obs = np.zeros((len(self.cfg.observation.camera_keys), 3*self.cfg.frame_stack, self.cfg.observation.camera_shape[0], self.cfg.observation.camera_shape[1]), dtype=np.uint8)
        self.stack_qpos = np.zeros((sum(item['dim'] for item in self.cfg.observation.low_dim_keys)*self.cfg.frame_stack,), dtype=np.float32)
        
        obs, _ = self.train_env.reset()
        terminated = False
        truncated = False
        rgb_obs, low_dim_obs = convert_obs(obs, self.cfg)
        
        # stack_rgb_obs, stack_low_dim_obs = self.update_frame_stack(rgb_obs, low_dim_obs, self.cfg.low_dim_obs_shape)
        for _ in range(self.cfg.frame_stack):
            stack_rgb_obs, stack_low_dim_obs = self.update_frame_stack(rgb_obs, low_dim_obs, sum(item['dim'] for item in self.cfg.observation.low_dim_keys))
        
        inst_samples = {
            'rgb_obs': rgb_obs,
            'qpos': low_dim_obs.astype(np.float32),
            'action': np.zeros(self.cfg.agent.action_shape).astype(np.float32),
            'reward': 0.0,
            'discount': 0.99,
            'demo': 0.0,
            'last': terminated or truncated,
        }
        
        self.replay_storage.add(inst_samples)
        self.demo_replay_storage.add(inst_samples)
        self.train_video_recorder.init(inst_samples["rgb_obs"][0])
        metrics = None
        while train_until_step(self.global_step):
            if inst_samples["last"]:
                self._global_episode += 1
                self.train_video_recorder.save(f"{self.global_frame}.mp4")
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(
                        self.global_frame, ty="train"
                    ) as log:
                        log("fps", episode_frame / elapsed_time)
                        log("total_time", total_time)
                        log("episode_reward", episode_reward)
                        log("episode_length", episode_frame)
                        log("episode", self.global_episode)
                        log("buffer_size", len(self.replay_storage))
                        log("demo_buffer_size", len(self.demo_replay_storage))
                        log("step", self.global_step)

                # do evaluation before resetting the environment
                if do_eval:
                    self.logger.log(
                        "eval_total_time", self.timer.total_time(), self.global_frame
                    )
                    self.eval()
                    do_eval = False

                # reset env
                self.stack_rgb_obs = np.zeros((len(self.cfg.observation.camera_keys), 3*self.cfg.frame_stack, self.cfg.observation.camera_shape[0], self.cfg.observation.camera_shape[1]), dtype=np.uint8)
                self.stack_qpos = np.zeros((sum(item['dim'] for item in self.cfg.observation.low_dim_keys)*self.cfg.frame_stack,), dtype=np.float32)
                
                obs, _ = self.train_env.reset()
                terminated = False
                truncated = False
                rgb_obs, low_dim_obs = convert_obs(obs, self.cfg)
                # stack_rgb_obs, stack_low_dim_obs = self.update_frame_stack(rgb_obs, low_dim_obs, self.cfg.low_dim_obs_shape)
                for _ in range(self.cfg.frame_stack):
                    stack_rgb_obs, stack_low_dim_obs = self.update_frame_stack(rgb_obs, low_dim_obs, sum(item['dim'] for item in self.cfg.observation.low_dim_keys))
                
                inst_samples = {
                    'rgb_obs': rgb_obs,
                    'qpos': low_dim_obs.astype(np.float32),
                    'action': np.zeros(self.cfg.agent.action_shape).astype(np.float32),
                    'reward': 0.0,
                    'discount': 0.99,
                    'demo': 0.0,
                    'last': terminated or truncated,
                }
                self.replay_storage.add(inst_samples)
                self.demo_replay_storage.add(inst_samples)
                self.train_video_recorder.init(inst_samples["rgb_obs"][0])
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # set a flag to initate evaluation when the current episode terminates
            if self.global_step >= self.cfg.eval_every_frames and eval_every_step(
                self.global_step
            ):
                do_eval = True

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(
                    torch.tensor(stack_rgb_obs).float(),
                    torch.tensor(stack_low_dim_obs),
                    self.global_step,
                    eval_mode=False,
                )

            # try to update the agent
            if not seed_until_step(self.global_step):
                for _ in range(self.cfg.num_update_steps):
                    metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty="train")

            # take env step
            obs, reward, terminated, truncated, info = self.train_env.step(action)
            rgb_obs, low_dim_obs = convert_obs(obs, self.cfg)
            is_success = info["success"]
            sparse_reward = 1.0 if is_success else 0.0
            stack_rgb_obs, stack_low_dim_obs = self.update_frame_stack(rgb_obs, low_dim_obs, sum(item['dim'] for item in self.cfg.observation.low_dim_keys))
            inst_samples = {
                'rgb_obs': rgb_obs,
                'qpos': low_dim_obs.astype(np.float32),
                'action': action,
                'reward': sparse_reward,
                'discount': 0.99,
                'demo': 0.0,
                'last': terminated or truncated,
            }
            episode_reward += reward
            self.replay_storage.add(inst_samples)
            self.demo_replay_storage.add(inst_samples)
            self.train_video_recorder.record(inst_samples["rgb_obs"][0])
            episode_step += 1
            self._global_step += 1
            
    def load_ms2_demos_from_h5(self, dataset_file):
            
        import h5py
        from mani_skill2.utils.io_utils import load_json
        from tqdm import tqdm
        
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]

        self.obs_state = []
        self.obs_rgbd = []
        self.actions = []
        self.total_frames = 0
        load_count = self.cfg.num_demos
        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)
            observations = trajectory["obs"]
            actions = trajectory["actions"]
            successes = trajectory["success"]
            
            length = len(observations["agent"]["qpos"])
            
            self.stack_rgb_obs = np.zeros((len(self.cfg.observation.camera_keys), 3*self.cfg.frame_stack, self.cfg.observation.camera_shape[0], self.cfg.observation.camera_shape[1]), dtype=np.uint8)
            self.stack_qpos = np.zeros((sum(item['dim'] for item in self.cfg.observation.low_dim_keys)*self.cfg.frame_stack,), dtype=np.float32)
            
            for i_traj in range(length):
                
                # image data is not scaled here and is kept as uint16 to save space
                rgb_b = cv2.resize(observations["image"]['base_camera']['rgb'][i_traj], (self.cfg.observation.camera_shape[0], self.cfg.observation.camera_shape[1])).astype(np.uint8)
                rgb_h = cv2.resize(observations["image"]['hand_camera']['rgb'][i_traj], (self.cfg.observation.camera_shape[0], self.cfg.observation.camera_shape[1])).astype(np.uint8)
                
                # transpose to (C, H, W)
                rgb_b = rgb_b.transpose(2, 0, 1)[np.newaxis]
                rgb_h = rgb_h.transpose(2, 0, 1)[np.newaxis]
                                
                rgb = np.concatenate([rgb_b, rgb_h], axis=0)
                
                if i_traj == length - 1:
                    terminated = True
                    truncated = True
                    reward = 1.0
                    action = actions[i_traj-1]
                elif i_traj == 0:
                    terminated = False
                    truncated = False
                    reward = 0.0
                    action = np.zeros(self.cfg.agent.action_shape).astype(np.float32)
                else:
                    terminated = False
                    truncated = False
                    reward = 0.0
                    action = actions[i_traj-1]
                
                low_dim_obs_list = []
                for item in self.cfg.observation.low_dim_keys:
                    key, expected_dim = item['key'], item['dim']
                    if key == "goal_relative":
                        value = observations["extra"]["goal_pos"][i_traj] - observations["extra"]["tcp_pose"][i_traj][:3]
                    elif key in observations["agent"]:
                        value = observations["agent"][key][i_traj]
                    elif key in observations["extra"]:
                        value = observations["extra"][key][i_traj]
                    else:
                        raise ValueError(f"Unknown key: {key}")
                    
                    if value.shape[-1] != expected_dim:
                        raise ValueError(f"Dimension mismatch for {key}: expected {expected_dim}, got {value.shape[-1]}")
                    
                    low_dim_obs_list.append(value)
                
                low_dim_obs = np.concatenate(low_dim_obs_list, axis=0).astype(np.float32)
                
                stack_rgb_obs, stack_low_dim_obs = self.update_frame_stack(rgb, low_dim_obs, sum(item['dim'] for item in self.cfg.observation.low_dim_keys))
                
                inst_samples = {
                    'rgb_obs': rgb,
                    'qpos': low_dim_obs,
                    'action': action,
                    'reward': reward,
                    'discount': 0.99,
                    'demo': 1.0,
                    'last': terminated or truncated,
                }
                
                self.replay_storage.add(inst_samples)
                self.demo_replay_storage.add(inst_samples)

    def load_ms2_demos(self):
        if self.cfg.num_demos > 0:
            base_path = hydra.utils.get_original_cwd()
            if isinstance(self.cfg.dataset_file, str):
                dataset_file = os.path.join(base_path, self.cfg.dataset_file)
                self.load_ms2_demos_from_h5(dataset_file)
            else:
                dataset_files = [os.path.join(base_path, f) for f in self.cfg.dataset_file]
                for dataset_file in dataset_files:
                    self.load_ms2_demos_from_h5(dataset_file)
        else:
            logging.warning("Not using demonstrations")

    def save_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        keys_to_save = ["agent", "timer", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        with snapshot.open("rb") as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path="cfgs", config_name="config_maniskill2")
def main(cfg):
    from train_ms2 import Workspace as W

    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / "snapshot.pt"
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot()
    workspace.load_ms2_demos()
    workspace.train()


if __name__ == "__main__":
    main()
