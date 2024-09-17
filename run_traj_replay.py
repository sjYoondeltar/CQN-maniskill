import subprocess
import sys

suffixes = [
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "021_bleach_cleanser"
]

# suffixes = []
# # 065 cup a부터 j까지 추가
# cups = [f"065-{chr(i)}_cup" for i in range(ord('a'), ord('j')+1)]
# suffixes.extend(cups)

for suffix in suffixes:
    command = [
        "python", "-m", "mani_skill2.trajectory.replay_trajectory",
        "--traj-path", f"demos/v0/rigid_body/PickSingleYCB-v0/{suffix}.h5",
        "--save-traj",
        "--obs-mode", "rgbd",
        "--target-control-mode", "pd_ee_delta_pose",
        "--num-procs", "10"
    ]
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while processing {suffix}: {e}", file=sys.stderr)
        sys.exit(1)

print("All trajectories processed successfully.")
