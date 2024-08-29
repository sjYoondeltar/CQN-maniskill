#!/bin/bash

for suffix in {a..j}
do
    python -m mani_skill2.trajectory.replay_trajectory --traj-path \
    demos/v0/rigid_body/PickSingleYCB-v0/065-${suffix}_cups.h5 --save-traj --obs-mode rgbd --target-control-mode pd_joint_delta_pos --num-procs 10
done
