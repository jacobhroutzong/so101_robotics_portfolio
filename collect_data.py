import numpy as np
from pathlib import Path
from robot_env import MujocoSimEnv

def collect_rollouts(num_episodes=5, max_steps=200, save_dir="data/raw_rollouts"):
    env = MujocoSimEnv()
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    all_observations = []
    all_actions = []
    all_rewards = []
    episode_indices = []
    
    print(f"Starting collection for {num_episodes} episodes ({max_steps} steps each)...")
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            # Generate a random action within control limits
            action = np.random.uniform(
                env.model.actuator_ctrlrange[:, 0],
                env.model.actuator_ctrlrange[:, 1]
            )
            
            next_obs = env.step(action)
            
            # Log transition data matching LeRobot schema conventions
            all_observations.append(obs["observation.state"])
            all_actions.append(action)
            all_rewards.append(0.0) # Placeholder reward for imitation learning
            episode_indices.append(ep)
            
            obs = next_obs
            step_count += 1
            
        print(f"Episode {ep + 1}/{num_episodes} completed ({step_count} steps).")
        
    # Save as compressed numpy archive (easily loaded into LeRobot dataset builders)
    data_file = save_path / "rollouts.npz"
    np.savez(
        data_file,
        observations=np.array(all_observations),
        actions=np.array(all_actions),
        rewards=np.array(all_rewards),
        episode_index=np.array(episode_indices)
    )
    print(f"Successfully saved rollouts to {data_file}")

if __name__ == "__main__":
    collect_rollouts()
