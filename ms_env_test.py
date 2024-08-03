import gymnasium as gym
import mani_skill2.envs
import cv2

env = gym.make("PickCube-v0", obs_mode="rgbd", control_mode="pd_joint_delta_pos", render_mode="cameras")
print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    render = env.render()  # a display is required to render
    # print("render", render)
    cv2.imshow("render", render)
    cv2.waitKey(10)
    
env.close()
