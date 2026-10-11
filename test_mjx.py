import os
import sys
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.4'

old_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp
sys.stderr.close()
sys.stderr = old_stderr

m = mujoco.MjModel.from_xml_path('arm.xml')
d = mujoco.MjData(m)

mx = mjx.put_model(m)
dx = mjx.put_data(m, d)

@jax.jit
def step_fn(data):
    data = data.replace(ctrl=jnp.array([0.5, -0.2]))
    return mjx.step(mx, data)

dx_res = step_fn(dx)
print("MJX GPU step successful! Joint positions:", dx_res.qpos)
