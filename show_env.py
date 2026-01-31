import sys
import torch
import jax
import jaxlib
import brax
import mujoco
import evox
import importlib.metadata
from brax import envs

# 1. Check Hardware and Core Frameworks
print("-" * 30)
print(f"Python executable: {sys.executable}")
print(f"Python version:    {sys.version.split()[0]}")
print(f"JAX version:       {jax.__version__}")
print(f"JAXLIB version:    {jaxlib.__version__}")
print(f"JAX Devices:       {jax.devices()}")  # Should show [CudaDevice(id=0)]

# 2. Check PyTorch and CUDA Support
print("-" * 30)
print(f"PyTorch version:   {torch.__version__}")
print(f"CUDA available:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device Name:   {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No CUDA device detected for PyTorch!")

# 3. Check Library Metadata and Locations
print("-" * 30)
print(f"EvoX location:     {evox.__file__}")
print(f"EvoX version:      {importlib.metadata.version('evox')}")
print(f"Brax version:      {importlib.metadata.version('brax')}")
print(f"MuJoCo version:    {importlib.metadata.version('mujoco')}")

# 4. Simulate Brax Environment Loading (Diagnostic)
print("-" * 30)
try:
    env_name = "swimmer"
    backend = "generalized"
    print(f"Loading '{env_name}' with '{backend}' backend...")

    # Initialize environment
    env = envs.get_environment(env_name=env_name, backend=backend)

    print("✅ Environment successfully loaded!")
    print(f"Observation size:  {env.observation_size}")
    print(f"Action size:       {env.action_size}")

except Exception as e:
    print(f"❌ Failed to initialize environment: {e}")
    import traceback

    traceback.print_exc()

print("-" * 30)