import os
import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Expose VLM4VLA to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "VLM4VLA"))

CALVIN_FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": None,
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (15,),
        "names": None,
    },
    "observation.images.static": {
        "dtype": "video",
        "shape": (200, 200, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.gripper": {
        "dtype": "video",
        "shape": (84, 84, 3),
        "names": ["height", "width", "channels"],
    },
    "task": {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    },
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin_dir", type=str, required=True, help="Path to calvin dataset (e.g. /path/to/task_ABC_D)")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save LeRobot dataset")
    parser.add_argument("--split", type=str, default="training", choices=["training", "validation"])
    parser.add_argument("--limit_episodes", type=int, default=-1, help="Limit number of episodes to process (for debugging)")
    args = parser.parse_args()

    calvin_dir = Path(args.calvin_dir)
    split_dir = calvin_dir / args.split
    
    # Load language annotations
    print(f"Loading auto_lang_ann.npy from {split_dir}...")
    try:
        lang_data = np.load(split_dir / "lang_annotations" / "auto_lang_ann.npy", allow_pickle=True).item()
    except Exception:
        lang_data = np.load(split_dir / "auto_lang_ann.npy", allow_pickle=True).item()
        
    ep_start_end_ids = lang_data["info"]["indx"]
    lang_ann = lang_data["language"]["ann"]
    
    # Initialize LeRobot dataset
    repo_id = f"local/calvin_{args.split}"
    root_dir = Path(args.output_dir) / args.split
    print(f"Initializing LeRobot dataset at {root_dir}...")
    
    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        features=CALVIN_FEATURES,
        root=root_dir,
    )
    
    num_episodes = len(ep_start_end_ids)
    if args.limit_episodes > 0:
        num_episodes = min(num_episodes, args.limit_episodes)
        
    print(f"Processing {num_episodes} episodes...")
    for i in tqdm(range(num_episodes)):
        start_idx, end_idx = ep_start_end_ids[i]
        instruction = lang_ann[i]
        
        for idx in range(start_idx, end_idx + 1):
            file_name = f"episode_{idx:07d}.npz"
            file_path = split_dir / file_name
            
            # Load frame
            frame_data = np.load(file_path)
            
            rgb_static = frame_data["rgb_static"] # (200, 200, 3)
            rgb_gripper = frame_data["rgb_gripper"] # (84, 84, 3)
            robot_obs = frame_data["robot_obs"] # (15,)
            rel_actions = frame_data["rel_actions"].copy() # (7,)
            
            # Normalize action gripper to [0, 1] range:
            # calvin relative action last dimension is gripper state (-1 to 1).
            rel_actions[-1] = (rel_actions[-1] + 1) / 2.0
            
            frame = {
                "action": torch.from_numpy(rel_actions).float(),
                "observation.state": torch.from_numpy(robot_obs).float(),
                "observation.images.static": torch.from_numpy(rgb_static),
                "observation.images.gripper": torch.from_numpy(rgb_gripper),
                "task": instruction,
            }
            
            lerobot_dataset.add_frame(frame)
            
        lerobot_dataset.save_episode()
        
    lerobot_dataset.finalize()
    print("Done!")

if __name__ == "__main__":
    main()
