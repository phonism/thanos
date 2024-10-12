import operator
import torch
import os
import math
from functools import reduce
import numpy as np
import thanos
from . import ndarray_backend_numpy
from . import ndarray_backend_cpu
#from .triton_ndarray import *

if os.getenv("NDARRAY_BACKEND") == "TRITON":
    from .triton_ndarray import *
elif os.getenv("NDARRAY_BACKEND") == "TORCH":
    from .pytorch_ndarray import *
else:
    from .cuda_ndarray import *


class BackendDevice:
    """A backend device, wraps the implementation module."""

    def __init__(self, name, mod):
        self.name = name
        self.mod = mod

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return self.name + "()"

    def __getattr__(self, name):
        return getattr(self.mod, name)

    def enabled(self):
        return self.mod is not None

    def randn(self, *shape, dtype="float32"):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return NDArray(torch.randn(*shape, device="cuda"), device=self)
        #return NDArray(np.random.randn(*shape).astype(dtype), device=self)

    def rand(self, *shape, dtype="float32"):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        # TODO 这里先用torch的rand
        return NDArray(torch.rand(*shape, device="cuda"), device=self)
        #return NDArray(np.random.rand(*shape).astype(dtype), device=self)

    def one_hot(self, n, i, dtype="float32"):
        return NDArray(np.eye(n, dtype=dtype)[np.array(i).astype(int)], device=self)

    def empty(self, shape, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        #assert dtype == "float32"
        return NDArray.make(shape, device=self)

    def full(self, shape, fill_value, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        #assert dtype == "float32"
        arr = self.empty(shape, dtype)
        arr.fill(fill_value)
        return arr


def cuda():
    """Return cuda device"""
    try:
        from . import ndarray_backend_cuda
        return BackendDevice("cuda", ndarray_backend_cuda)
    except ImportError:
        return BackendDevice("cuda", None)

def triton():
    """Return cuda device"""
    try:
        from .triton_ndarray import NDArray
        return BackendDevice("cuda", ndarray_backend_cuda)
    except ImportError:
        return NDArray


def cpu_numpy():
    """Return numpy device"""
    return BackendDevice("cpu_numpy", ndarray_backend_numpy)


def cpu():
    """Return cpu device"""
    return BackendDevice("cpu", ndarray_backend_cpu)



def default_device():
    return cpu()


def all_devices():
    """return a list of all available devices"""
    return [cpu(), cuda(), cpu_numpy()]


def array(a, dtype="float32", device=None):
    """ Convenience methods to match numpy a bit more closely."""
    dtype = "float32" if dtype is None else dtype
    #assert dtype == "float32"
    return NDArray(a, device=device, dtype=dtype)


def empty(shape, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return devie.empty(shape, dtype)


def full(shape, fill_value, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.full(shape, fill_value, dtype)


def broadcast_to(array, new_shape):
    return array.broadcast_to(new_shape)

def swapaxes(array, x, y):
    new_shape = list(range(len(array.shape)))
    if x < 0:
        x = x + len(new_shape)
    if y < 0:
        y = y + len(new_shape)
    new_shape[x] = y
    new_shape[y] = x
    return array.permute(tuple(new_shape))

def norm_axis(a, axis):
    if type(axis) is int:
        axis = (axis,)
    new_axis = []
    for ax in axis:
        if ax < 0:
            ax = ax + len(a.shape)
        new_axis.append(ax)
    return tuple(new_axis)

def sum(a, axis=None, keepdims=False):
    if type(axis) is int:
        axis = (axis, )
    if axis is None:
        return a.sum(axis=axis, keepdims=keepdims)
    axis = norm_axis(a, axis)
    axis = tuple(sorted(list(axis)))
    pre = 0
    for ax in axis:
        if keepdims:
            a = a.sum(axis=ax, keepdims=keepdims)
        else:
            a = a.sum(axis=ax - pre, keepdims=keepdims)
        pre += 1
    return a

def max(a, axis=None, keepdims=False):
    if type(axis) is int:
        axis = (axis, )
    if axis is None:
        return a.max(axis=axis, keepdims=keepdims)
    axis = norm_axis(a, axis)
    axis = tuple(sorted(list(axis)))
    pre = 0
    for ax in axis:
        if keepdims:
            a = a.max(axis=ax, keepdims=keepdims)
        else:
            a = a.max(axis=ax - pre, keepdims=keepdims)
        pre += 1
    return a

def reshape(array, new_shape):
    return array.reshape(new_shape)

def negative(array):
    return -array

def divide(a, b, dtype):
    return a / b

def power(a, b):
    return a ** b

def sin(a):
    return a.sin()

def cos(a):
    return a.cos()

def log(a):
    return a.log()

def exp(a):
    return a.exp()

def matmul(a, b):
    return a.matmul(b)

def maximum(a, b):
    return a.maximum(b)

def diag(a):
    return a.diag()

def triu(a, k=0):
    return a.triu(k=k)

def sqrt(a):
    return a.sqrt()
