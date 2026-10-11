import mujoco
import mujoco.viewer
import time

m = mujoco.MjModel.from_xml_path('arm.xml')
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(0.01)
