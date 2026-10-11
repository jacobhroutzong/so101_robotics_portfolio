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
    def __init__(self, xml_path="arm.xml"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.get_observation()
        
    def step(self, action):
        # Clip action to control limits
        self.data.ctrl[:] = np.clip(action, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        mujoco.mj_step(self.model, self.data)
        return self.get_observation()
        
    def get_observation(self):
        # Match LeRobot schema conventions: dictionary of states
        return {
            "observation.state": np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()]),
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy()
        }

# Example validation test
if __name__ == "__main__":
    env = MujocoSimEnv()
    obs = env.reset()
    print("Initial observation keys:", obs.keys())
    print("Initial state shape:", obs["observation.state"].shape)
    
    next_obs = env.step(np.zeros(env.model.nu))
    print("Step successful! New joint positions:", next_obs["qpos"])
