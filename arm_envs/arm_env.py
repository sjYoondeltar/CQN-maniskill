
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Rectangle
import matplotlib as mpl
import sys
from arm_envs.sensor_model import SimpleSENSOR

class ArmEnv(object):
    action_bound = [-2, 2]
    action_dim = 2
    state_dim = 7
    dt = .1  # refresh rate
    arm1l = 100
    arm2l = 100
    viewer = None
    viewer_xy = (400, 400)
    get_point = False
    mouse_in = np.array([False])
    point_l = 10
    grab_counter = 0

    def __init__(self, ):
        # node1 (l, d_rad, x, y),
        # node2 (l, d_rad, x, y)
        self.arm_info = np.zeros((2, 4))
        self.arm_info[0, 0] = self.arm1l
        self.arm_info[1, 0] = self.arm2l
        
        self.renew_obj()
        
        self.center_coord = np.array(self.viewer_xy)/2
        
        self.sensor = SimpleSENSOR(
            sensor_max=200,
            n_sensor=51,
            range_sensor=[-np.pi/2, np.pi/2]
        )
        
        self.bound_pts = np.array([
            [0, 0],
            [400, 0],
            [400, 400],
            [0, 400]
        ])


    def step(self, action):
        # action = (node1 angular v, node2 angular v)
        action = np.clip(action, *self.action_bound)
        self.arm_info[:, 1] += action * self.dt
        self.arm_info[:, 1] %= np.pi * 2
        # self.arm_info[1, 1] = np.clip(self.arm_info[1, 1], 0, np.pi)

        arm1rad = self.arm_info[0, 1]
        arm2rad = self.arm_info[1, 1]
        arm1dx_dy = np.array([self.arm_info[0, 0] * np.cos(arm1rad), self.arm_info[0, 0] * np.sin(arm1rad)])
        arm2dx_dy = np.array([self.arm_info[1, 0] * np.cos(arm1rad + arm2rad), self.arm_info[1, 0] * np.sin(arm1rad + arm2rad)])
        self.arm_info[0, 2:4] = self.center_coord + arm1dx_dy  # (x1, y1)
        self.arm_info[1, 2:4] = self.arm_info[0, 2:4] + arm2dx_dy  # (x2, y2)

        sensor_dis = self._get_state()
        arm_connect = self.arm_info[0, 2:4]
        arm_end = self.arm_info[1, 2:4]
        self.update_obj()
        r = self._r_func()

        return sensor_dis, arm_connect, arm_end, r, self.get_point
    
    def renew_obj(self, fix_obj_list=None):
        
        if fix_obj_list:
            self.obj_info = np.array([
                fix_obj_list
            ])
            
        else:
            mag1 = np.random.rand(1)[0] * 150 + 50
            angle1 = np.random.rand(1)[0] * np.pi
            mag2 = np.random.rand(1)[0] * 150 + 50
            angle2 = np.random.rand(1)[0] * np.pi
            self.obj_info = np.asarray([
                [mag1*np.cos(angle1)+200, mag1*np.sin(angle1) + 200, 20, 20, 45*np.random.rand(1)[0], 0, 0],
                # [mag2*np.cos(angle2)+200, mag2*np.sin(angle2) + 200, 20, 20, 45*np.random.rand(1)[0], 0, 1],
                # [400*np.random.rand(1), 400*np.random.rand(1), 20, 20, 45*np.random.rand(1)],
                # [400*np.random.rand(1), 400*np.random.rand(1), 20, 20, 45*np.random.rand(1)]
            ], dtype=object)
        
    def update_obj(self):
        
        arm_end = self.arm_info[1, 2:4]
        
        for obj_ist in self.obj_info.tolist():
            
            if np.linalg.norm(obj_ist[:2] - arm_end) < self.point_l and obj_ist[5] == 0  and obj_ist[6] == 0:
                self.get_point = False
                if self.grab_counter == 10:
                    self.get_obj = True
                self.grab_counter += 1
                # obj_ist[5] = 1
                # obj_ist[:2] = arm_end
                break
            
            elif self.get_obj and obj_ist[5] == 1:
                obj_ist[:2] = arm_end
                
            else:
                pass
        
    def reset(self):
        self.get_point = False
        self.get_obj = False
        self.grab_counter = 0
        
        # arm1rad = np.pi/2
        arm1rad = np.random.rand(1)[0] * np.pi
        arm1rad %= np.pi * 2
        arm2rad = np.pi/2
        # arm2rad = np.random.rand(1)[0] * np.pi
        arm2rad %= np.pi * 2
        self.arm_info[0, 1] = arm1rad
        self.arm_info[1, 1] = arm2rad
        arm1dx_dy = np.array([self.arm_info[0, 0] * np.cos(arm1rad), self.arm_info[0, 0] * np.sin(arm1rad)])
        arm2dx_dy = np.array([self.arm_info[1, 0] * np.cos(arm1rad + arm2rad), self.arm_info[1, 0] * np.sin(arm1rad + arm2rad)])
        self.arm_info[0, 2:4] = self.center_coord + arm1dx_dy  # (x1, y1)
        self.arm_info[1, 2:4] = self.arm_info[0, 2:4] + arm2dx_dy  # (x2, y2)
        
        self.renew_obj()
        
        sensor_dis = self._get_state()
        arm_connect = self.arm_info[0, 2:4]
        arm_end = self.arm_info[1, 2:4]
        
        return sensor_dis, arm_connect, arm_end

    def render(self):
        
        fig = plt.figure(1, figsize=[7, 7])

        ax = fig.add_subplot(1,1,1)
        
        ## Plot the arm
        ax.plot([self.center_coord[0], self.arm_info[0, 2]], [self.center_coord[1], self.arm_info[0, 3]], 'b-', linewidth=5)
        ax.plot([self.arm_info[0, 2], self.arm_info[1, 2]], [self.arm_info[0, 3], self.arm_info[1, 3]], 'b-', linewidth=5)
        
        ## Plot the sensor
        for s_idx in range(self.sensor.n_sensor):
            ax.plot([self.center_coord[0], self.sensor.sensor_info[s_idx, 1]], [self.center_coord[1], self.sensor.sensor_info[s_idx, 2]], 'r-', linewidth=1)
        
        ## Plot the objects
        
        for idx_obj, obj in enumerate(self.obj_info.tolist()):
            
            v_sur = Rectangle(
                (-obj[2]/2, -obj[3]/2),
                obj[2],
                obj[3],
                0,
                edgecolor='red',
                facecolor='green',
                fill=True
            )
            
            t2 = mpl.transforms.Affine2D().rotate_deg(obj[4]) + \
                mpl.transforms.Affine2D().translate(obj[0], obj[1])+ ax.transData
            
            v_sur.set_transform(t2)
            
            ax.add_patch(v_sur)
        
        plt.axis('equal')

        ax.set(title='Arm', xlabel='xAxis', ylabel='yAxis')
        ax.set_xlim([0, 400])
        ax.set_ylim([0, 400])
        
        plt.draw()

        plt.pause(0.01)
        
        fig.clf()

    def sample_action(self):
        return np.random.uniform(*self.action_bound, size=self.action_dim)

    def _get_state(self):
        # return the distance (dx, dy) between arm finger point with blue point
        
        obs_pts = np.array([[
            [obs[0]-obs[2]/2, obs[1]-obs[3]/2],
            [obs[0]+obs[2]/2, obs[1]-obs[3]/2],
            [obs[0]+obs[2]/2, obs[1]+obs[3]/2],
            [obs[0]-obs[2]/2, obs[1]+obs[3]/2],
        ] for obs in self.obj_info])
        
        self.sensor.update_sensors(np.asarray([self.center_coord[0], self.center_coord[1], np.pi/2]), obs_pts, self.bound_pts)
        
        sensor_measure = self.sensor.sensor_info[:, 0].reshape([-1, 1])/self.sensor.sensor_max
        
        return sensor_measure

    def _r_func(self):
        
        arm_end = self.arm_info[1, 2:4]
        
        if np.linalg.norm(self.obj_info[0, :2] - arm_end) < self.point_l:
            
            return 1
        
        else:
            
            r = -0.1*(np.linalg.norm(self.obj_info[0, :2] - arm_end)/200)**2
            
            return r


if __name__ == '__main__':
    
    arm_env_tmp = ArmEnv()
    
    for _ in range(20):
        
        obs, arm_end = arm_env_tmp.reset()
        
        for _ in range(10):
            
            obs, arm_end, r, done = arm_env_tmp.step(arm_env_tmp.sample_action())
            
            arm_env_tmp.render()
            
            time.sleep(0.1)
