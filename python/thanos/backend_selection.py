"""Logic for backend selection"""
import os

BACKEND = os.environ.get("THANOS_BACKEND", "thanos")

if BACKEND == "thanos":
    if "NDARRAY_BACKEND" not in os.environ:
        os.environ["NDARRAY_BACKEND"] = "TRITON"
    from . import backend_ndarray as array_api
    from .backend_ndarray import (
        all_devices,
        cuda,
        triton,
        cpu,
        cpu_numpy,
        default_device,
        BackendDevice as Device,
    )
    NDArray = array_api.NDArray
elif BACKEND == "numpy":
    import numpy as array_api
    from .backend_numpy import all_devices, cpu, default_device, Device

    NDArray = array_api.ndarray
else:
    raise RuntimeError("Unknown needle array backend %s" % BACKEND)
