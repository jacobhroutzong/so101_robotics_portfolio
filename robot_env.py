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
    def __init__(self, xml_path="arm.xml", latency_steps=2, noise_std=0.005, width=320, height=240):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Sim-to-Real parameters
        self.latency_steps = latency_steps
        self.noise_std = noise_std
        self.action_buffer = []
        
        # Initialize MuJoCo offscreen renderer for visual imitation learning (LeRobot standard)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        
        mujoco.mj_resetData(self.model, self.data)
        
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        self.action_buffer = [np.zeros(self.model.nu)] * self.latency_steps
        return self.get_observation()
        
    def step(self, action):
        clipped_action = np.clip(action, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        
        # Simulate serial bus command latency
        self.action_buffer.append(clipped_action)
        delayed_action = self.action_buffer.pop(0)
        
        self.data.ctrl[:] = delayed_action
        mujoco.mj_step(self.model, self.data)
        
        return self.get_observation()
        
    def get_observation(self):
        # Inject minor sensor encoder noise
        qpos_noisy = self.data.qpos.copy() + np.random.normal(0, self.noise_std, size=self.data.qpos.shape)
        qvel_noisy = self.data.qvel.copy() + np.random.normal(0, self.noise_std, size=self.data.qvel.shape)
        
        # Render offscreen camera frame (LeRobot image key convention)
        self.renderer.update_scene(self.data)
        rgb_image = self.renderer.render() # Shape: (height, width, 3), uint8
        
        return {
            "observation.state": np.concatenate([qpos_noisy, qvel_noisy]),
            "observation.images.front": rgb_image,
            "qpos": qpos_noisy,
            "qvel": qvel_noisy,
            "qpos_clean": self.data.qpos.copy()
        }

if __name__ == "__main__":
    env = MujocoSimEnv()
    obs = env.reset()
    print("Environment initialized with offscreen camera rendering.")
    print("Observation state shape:", obs["observation.state"].shape)
    print("Camera frame shape (LeRobot format):", obs["observation.images.front"].shape)
    
    next_obs = env.step(np.zeros(env.model.nu))
    print("Step successful with image capture!")
