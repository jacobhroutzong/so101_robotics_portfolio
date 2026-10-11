import mujoco
import numpy as np

class BaseRobotEnv:
    def reset(self):
        raise NotImplementedError
    def step(self, action):
        raise NotImplementedError
    def get_observation(self):
        raise NotImplementedError

class MujocoSimEnv(BaseRobotEnv):
    def __init__(self, xml_path="arm.xml", latency_steps=2, noise_std=0.005):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Sim-to-Real parameters tailored for serial bus servo arms (like SO-101)
        self.latency_steps = latency_steps
        self.noise_std = noise_std
        self.action_buffer = []
        
        mujoco.mj_resetData(self.model, self.data)
        
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        # Clear action history buffer on reset
        self.action_buffer = [np.zeros(self.model.nu)] * self.latency_steps
        return self.get_observation()
        
    def step(self, action):
        # Clip action to control limits
        clipped_action = np.clip(action, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        
        # Simulate serial bus command transmission latency by buffering actions
        self.action_buffer.append(clipped_action)
        delayed_action = self.action_buffer.pop(0)
        
        # Apply delayed action to model controls
        self.data.ctrl[:] = delayed_action
        mujoco.mj_step(self.model, self.data)
        
        return self.get_observation()
        
    def get_observation(self):
        # Inject minor sensor encoder noise to prevent over-fitting to clean sim data
        qpos_noisy = self.data.qpos.copy() + np.random.normal(0, self.noise_std, size=self.data.qpos.shape)
        qvel_noisy = self.data.qvel.copy() + np.random.normal(0, self.noise_std, size=self.data.qvel.shape)
        
        return {
            "observation.state": np.concatenate([qpos_noisy, qvel_noisy]),
            "qpos": qpos_noisy,
            "qvel": qvel_noisy,
            "qpos_clean": self.data.qpos.copy()
        }

if __name__ == "__main__":
    env = MujocoSimEnv()
    obs = env.reset()
    print("Sim-to-Real environment initialized successfully.")
    print("Observation state shape with noise injected:", obs["observation.state"].shape)
    
    next_obs = env.step(np.zeros(env.model.nu))
    print("Step executed with latency buffer and noise. Clean qpos:", next_obs["qpos_clean"])
