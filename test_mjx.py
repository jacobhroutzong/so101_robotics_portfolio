import os
# Limit JAX to use 40% of VRAM so it doesn't trigger OOM on your RTX 4070
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.4'

import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp

# 1. Load standard model and data
m = mujoco.MjModel.from_xml_path('arm.xml')
d = mujoco.MjData(m)

print("Default JAX devices:", jax.devices())

# 2. Put model and data onto the GPU via MJX
mx = mjx.put_model(m)
dx = mjx.put_data(m, d)

# 3. Define a JIT-compiled step function for maximum GPU throughput
@jax.jit
def step_fn(data):
    # Apply a control vector
    data = data.replace(ctrl=jnp.array([0.5, -0.2]))
    # Step the simulation forward using MJX
    return mjx.step(mx, data)

# 4. Run a quick JIT-compiled step batch
dx_res = step_fn(dx)
print("MJX GPU step successful! Joint positions:", dx_res.qpos)
