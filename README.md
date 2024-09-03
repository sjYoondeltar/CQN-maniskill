# CQN-maniskill

This repository is the simple application of CQN agent for the ManiSkill2 Environment

## How to set Maniskill2 demonstration data

To download the asset and demo datasets:
    
    # Download the asset
    python -m mani_skill2.utils.download_asset PickSingleYCB-v0

    # Download the demonstration dataset for a specific task
    python -m mani_skill2.utils.download_demo PickCube-v0
    python -m mani_skill2.utils.download_demo PickSingleYCB-v0
    python -m mani_skill2.utils.download_demo PegInsertionSide-v0
    python -m mani_skill2.utils.download_demo PlugCharger-v0
    python -m mani_skill2.utils.download_demo StackCube-v0

To convert the demo datasets to the rgbd image mode:

    python -m mani_skill2.trajectory.replay_trajectory --traj-path \
    demos/v0/rigid_body/PickCube-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_ee_delta_pose --num-procs 10

    python -m mani_skill2.trajectory.replay_trajectory --traj-path \
    demos/v0/rigid_body/PegInsertionSide-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_ee_delta_pose --num-procs 10

    python -m mani_skill2.trajectory.replay_trajectory --traj-path \
    demos/v0/rigid_body/StackCube-v0/trajectory.h5 --save-traj --obs-mode rgbd --target-control-mode pd_ee_delta_pose --num-procs 10

    # For the task PickSingleYCB-v0 with cups
    chmod +x run_traj_replay.sh
    ./run_traj_replay.sh

The control mode can be changed to `pd_joint_delta_pos` or `pd_ee_delta_pose` depending on the task. The number of processes can be adjusted according to the number of CPU cores available.

## How to train a model

To train a model for a specific task, please modify 'ms2_task@_global_' in "cfgs/config_maniskill2.yaml"
The following is an example of how to train a model for the task `PickSingleYCB-v0`:

```yaml
defaults:
  - _self_
  - ms2_task@_global_: picksingleycb_v0
```

Other task yaml files can be found in `cfgs/ms2_task`.

To train a model:

```python
    python train_ms2.py
```

### Single YCB Object Pick and Place results

![Single Cup Pick](media/single_cup_pick.gif)

### Reference

[***Continuous Control with Coarse-to-fine Reinforcement Learning***](https://younggyo.me/cqn/)

[***ManiSkill2***](https://github.com/haosulab/ManiSkill2-task-dev)
