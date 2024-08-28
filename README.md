# CQN-maniskill

### How to set Maniskill2 demonstration data

To download the asset and demo datasets:
    
    # Download the asset
    python -m mani_skill2.utils.download_asset PickSingleYCB-v0

    # Download the demonstration dataset for a specific task
    python -m mani_skill2.utils.download_demo PickCube-v0
    python -m mani_skill2.utils.download_demo PickSingleYCB-v0
    python -m mani_skill2.utils.download_demo PegInsertionSide-v0
    python -m mani_skill2.utils.download_demo PlugCharger-v0

To convert the demo datasets to the rgbd image mode:
    
    python -m mani_skill2.trajectory.replay_trajectory --traj-path \
    demos/v0/rigid_body/PickSingleYCB-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_ee_delta_pose --num-procs 10

    python -m mani_skill2.trajectory.replay_trajectory --traj-path \
    demos/v0/rigid_body/PickCube-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_ee_delta_pose --num-procs 10

    python -m mani_skill2.trajectory.replay_trajectory --traj-path \
    demos/v0/rigid_body/PegInsertionSide-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_ee_delta_pose --num-procs 10

The control mode can be changed to `pd_joint_delta_pos` or `pd_ee_delta_pose` depending on the task. The number of processes can be adjusted according to the number of CPU cores available.
