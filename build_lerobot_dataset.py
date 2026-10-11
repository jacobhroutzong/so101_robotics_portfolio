import numpy as np
from pathlib import Path
import json

def convert_to_lerobot_format(raw_data_path="data/raw_rollouts/rollouts.npz", output_dir="data/lerobot_sim_dataset"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    if not Path(raw_data_path).exists():
        print(f"Error: {raw_data_path} not found. Run collect_data.py first!")
        return
        
    data = np.load(raw_data_path)
    observations = data["observations"]
    actions = data["actions"]
    episode_indices = data["episode_index"]
    
    # Create metadata structure aligned with LeRobot dataset specs
    meta = {
        "total_episodes": int(np.max(episode_indices) + 1),
        "total_frames": len(observations),
        "fps": 30,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [observations.shape[1]]},
            "action": {"dtype": "float32", "shape": [actions.shape[1]]}
        }
    }
    
    with open(out_path / "meta.json", "w") as f:
        json.dump(meta, f, indent=4)
        
    # Save processed arrays ready for LeRobot data loader ingestion
    np.savez_compressed(
        out_path / "episode_data.npz",
        observations=observations,
        actions=actions,
        episode_indices=episode_indices
    )
    
    print(f"LeRobot-compatible dataset generated at: {output_dir}/")
    print(f"-> Total frames packed: {len(observations)}")

if __name__ == "__main__":
    convert_to_lerobot_format()
