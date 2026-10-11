import mujoco
from mujoco import mjx
import jax
import torch

print("-" * 40)
print("🔍 ENVIRONMENT VERIFICATION")
print("-" * 40)
print(f"MuJoCo Version: {mujoco.__version__}")
print(f"JAX Devices: {jax.devices()}")
print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch Device Name: {torch.cuda.get_device_name(0)}")
print("-" * 40)
