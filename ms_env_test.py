import gymnasium as gym
import mani_skill2.envs
import cv2
import numpy as np

def main_loop():

    env = gym.make("PickCube-v0", obs_mode="rgbd", control_mode="pd_joint_delta_pos", render_mode="cameras")
    print("Observation space", env.observation_space)
    print("Action space", env.action_space)

    obs, _ = env.reset()
    done = False
    for i in range(200):
        # action = env.action_space.sample()
        if i < 100:
            action = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        else:
            action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        obs, reward, terminated, truncated, info = env.step(action)
        render = env.render()  # a display is required to render
        # print("render", render)
        cv2.imshow("render", render)
        cv2.waitKey(10)
        
    env.close()


if __name__ == "__main__":
    main_loop()
