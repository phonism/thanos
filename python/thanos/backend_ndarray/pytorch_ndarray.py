import torch
torch.set_printoptions(precision=8)

import numpy as np
import time

import triton
import triton.language as tl

import operator
from functools import reduce

def prod(x):
    return reduce(operator.mul, x, 1)

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]



def get_autotune_config():
    return get_cuda_autotune_config()

class NDArray:
    def __init__(self, data, device=None, dtype=None):
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data)
            self.data = self.data.cuda()
            self._device = device
        elif isinstance(data, NDArray):
            self.data = data.data
            self._device = data.device
        elif isinstance(data, torch.Tensor):
            self.data = data
            self._device = device
        else:
            self.data = torch.from_numpy(np.array(data))
            self._device = device

    @staticmethod
    def make(shape, device=None):
        data = torch.empty(shape, device=torch.device("cuda"), dtype=torch.float32)
        return NDArray(data, device=device)

    def compact(self):
        return NDArray(self.data.contiguous())

    def fill(self, value):
        """ Fill (in place) with a constant value. """
        self.data.fill_(value)

    def __repr__(self):
        return "NDArray:" + self.data.__repr__()

    def __str__(self):
        return self.__repr__()

    def numpy(self):
        return self.data.cpu().numpy()

    def to_numpy(self):
        return self.data.copy_to_host()

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return self.data.shape

    @property
    def stride(self):
        return self.data.stride()

    def stride(self, dim):
        return self.data.stride(dim)

    @property
    def device(self):
        return self._device

    @property
    def size(self):
        return prod(self.shape)

    def triu(self, k=0):
        return NDArray(torch.triu(self.data, diagonal=k))

    @property
    def flat(self):
        return NDArray(self.data.reshape((self.size,)))

    def __add__(self, y):
        if isinstance(y, NDArray):
            return NDArray(self.data + y.data)
        return NDArray(self.data + y)

    __radd__ = __add__

    def __mul__(self, y):
        if isinstance(y, NDArray):
            return NDArray(self.data * y.data)
        return NDArray(self.data * y)

    __rmul__ = __mul__

    def __truediv__(self, y):
        if isinstance(y, NDArray):
            return NDArray(self.data / y.data)
        return NDArray(self.data / y)

    def maximum(self, y):
        if isinstance(y, NDArray):
            return NDArray(torch.maximum(self.data, y.data))
        y = torch.full_like(self.data, y)
        return NDArray(torch.maximum(self.data, y))

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, scalar):
        return NDArray(self.data ** scalar)

    def log(self):
        return NDArray(torch.log(self.data))

    def exp(self):
        return NDArray(torch.exp(self.data))

    def cos(self):
        return NDArray(torch.cos(self.data))

    def sin(self):
        return NDArray(torch.sin(self.data))

    def sqrt(self):
        return NDArray(torch.sqrt(self.data))

    def reshape(self, new_shape):
        out_tensor = NDArray(self.data.reshape(new_shape))
        return out_tensor

    def permute(self, new_axis):
        out_tensor = NDArray(self.data.permute(new_axis))
        return out_tensor

    def broadcast_to(self, new_shape):
        out_tensor = NDArray(self.data.broadcast_to(new_shape))
        return out_tensor

    def __getitem__(self, idxs):
        out_tensor = NDArray(self.data.__getitem__(idxs))
        return out_tensor

    def __setitem__(self, idxs, other):
        self.data.__setitem__(idxs, other.data)
        return self

    def __eq__(self, other):
        if isinstance(other, NDArray):
            out_tensor = NDArray(self.data.__eq__(other.data))
        else:
            out_tensor = NDArray(self.data.__eq__(other))
        return out_tensor

    def __ge__(self, other):
        if isinstance(other, NDArray):
            out_tensor = NDArray(self.data.__ge__(other.data))
        else:
            out_tensor = NDArray(self.data.__ge__(other))
        return out_tensor

    def __ne__(self, other):
        return 1 - (self == other)

    def __gt__(self, other):
        return (self >= other) * (self != other)

    def __lt__(self, other):
        return 1 - (self >= other)

    def __le__(self, other):
        return 1 - (self > other)

    def __matmul__(self, b, activation=""):
        # Check constraints.
        return NDArray(self.data @ b.data)

    def sum(self, axis=None, keepdims=False):
        return NDArray(torch.sum(self.data, axis=axis, keepdims=keepdims))

    def max(self, axis=None, keepdims=False):
        if axis is None:
            return NDArray(torch.max(self.data))
        if isinstance(axis, tuple):
            axis = axis[0]
        return NDArray(torch.max(self.data, dim=axis, keepdim=keepdims).values)
