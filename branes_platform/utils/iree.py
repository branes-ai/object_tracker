import iree.runtime as rt
import numpy as np
import torch
import iree.turbine.runtime as trt

def load_vmfb(path: str):
    with open(path, 'rb') as f:
        model = rt.load_vm_flatbuffer(f.read(), driver="local-task")
    return model

def numpy_to_buffer(arr: np.ndarray, device: str = "cpu"):
    """Wrap a NumPy array as a Turbine device buffer."""
    return trt.asdevicearray(device, arr)