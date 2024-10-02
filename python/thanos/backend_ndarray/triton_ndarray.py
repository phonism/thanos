import torch
import numpy as np
import time

import triton
import triton.language as tl
from .ndarray_backend_triton import CudaArray
#from ndarray_backend_triton import CudaArray

import operator
from functools import reduce

def prod(x):
    """
    prod
    """
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

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def add_scalar_kernel(x_ptr, scalar, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x + scalar
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def mul_scalar_kernel(x_ptr, scalar, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * scalar
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def mul_kernel(x_ptr, y_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def div_scalar_kernel(x_ptr, scalar, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x / scalar
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def div_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x / y
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def maximum_scalar_kernel(x_ptr, scalar, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.maximum(x, scalar)
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def maximum_kernel(x_ptr, y_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = tl.maximum(x, y)
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def log_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.log(x)
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def exp_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.exp(x)
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def cos_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.cos(x)
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def sin_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.sin(x)
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def sqrt_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.sqrt(x)
    tl.store(output_ptr + offsets, output, mask=mask)

#@triton.autotune(
        #configs=get_autotune_config(),
        #key=['M', 'N', 'K'],
#)
@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        ACTIVATION: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator,  allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

@triton.jit
def sum_kernel(x_ptr, output_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    offset = m_offset[:, None] * N + n_offset[None, :]
    m_mask = m_offset < M
    n_mask = n_offset < N
    mask = m_mask[:, None] & n_mask[None, :]
    inp = tl.load(x_ptr + offset, mask=mask, other=0)
    out = tl.sum(inp, axis=1)
    tl.store(output_ptr + m_offset, out, mask=m_mask)

@triton.jit
def max_kernel(x_ptr, output_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    offset = m_offset[:, None] * N + n_offset[None, :]
    m_mask = m_offset < M
    n_mask = n_offset < N
    mask = m_mask[:, None] & n_mask[None, :]
    inp = tl.load(x_ptr + offset, mask=mask, other=-float('inf'))
    out = tl.max(inp, axis=1)
    tl.store(output_ptr + m_offset, out, mask=m_mask)

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
        out_tensor = NDArray(self.data.reshape((self.size,)))
        return out_tensor

    def __add__(self, y):
        output = torch.empty_like(self.data, device=torch.device("cuda"), dtype=self.dtype)
        assert self.data.is_cuda and output.is_cuda
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        if isinstance(y, NDArray):
            if y.data.is_contiguous() is False:
                y.data = y.data.contiguous()
            add_kernel[grid](self.data, y.data, output.data, n_elements, BLOCK_SIZE=1024)
        else:
            add_scalar_kernel[grid](self.data, y, output.data, n_elements, BLOCK_SIZE=1024)
        return NDArray(output)

    __radd__ = __add__

    def __mul__(self, y):
        output = torch.empty_like(self.data, device=torch.device("cuda"), dtype=self.dtype)
        assert self.data.is_cuda and output.is_cuda
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), 1, 1)
        if isinstance(y, NDArray):
            if y.data.is_contiguous() is False:
                y.data = y.data.contiguous()
            mul_kernel[grid](self.data, y.data, output, n_elements, BLOCK_SIZE=1024)
        else:
            mul_scalar_kernel[grid](self.data, y, output, n_elements, BLOCK_SIZE=1024)
        return NDArray(output)

    __rmul__ = __mul__

    def __truediv__(self, y):
        output = torch.empty_like(self.data, dtype=self.dtype)
        assert self.data.is_cuda and output.is_cuda
        # TODO 这里为啥需要这么做?
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        if isinstance(y, NDArray):
            if y.data.is_contiguous() is False:
                y.data = y.data.contiguous()
            div_kernel[grid](self.data, y.data, output, n_elements, BLOCK_SIZE=1024)
        else:
            div_scalar_kernel[grid](self.data, y, output, n_elements, BLOCK_SIZE=1024)
        return NDArray(output)

    def maximum(self, y):
        output = torch.zeros(self.shape, device=torch.device("cuda"), dtype=self.dtype)
        assert self.data.is_cuda and output.is_cuda
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), 1, 1)
        if isinstance(y, NDArray):
            if y.data.is_contiguous() is False:
                y.data = y.data.contiguous()
            maximum_kernel[grid](self.data, y.data, output, n_elements, BLOCK_SIZE=1024)
        else:
            maximum_scalar_kernel[grid](self.data, y, output, n_elements, BLOCK_SIZE=1024)
        return NDArray(output)
    
    def __neg__(self):
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        return self * (-1)

    def __sub__(self, other):
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        if other.data.is_contiguous() is False:
            other.data = other.data.contiguous()
        return self + (-other)

    def __rsub__(self, other):
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        if isinstance(other, NDArray):
            if other.data.is_contiguous() is False:
                other.data = other.data.contiguous()
        return other + (-self)

    def __pow__(self, scalar):
        output = torch.empty_like(self.data, dtype=self.dtype)
        assert self.data.is_cuda and output.is_cuda
        # TODO 这里为啥需要这么做?
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        return NDArray(self.data ** scalar)

    def log(self):
        output = torch.empty_like(self.data, dtype=self.dtype)
        assert self.data.is_cuda and output.is_cuda
        # TODO 这里为啥需要这么做?
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        log_kernel[grid](self.data, output, n_elements, BLOCK_SIZE=1024)
        return NDArray(output)

    def exp(self):
        output = torch.empty_like(self.data, dtype=self.dtype)
        assert self.data.is_cuda and output.is_cuda
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        exp_kernel[grid](self.data, output, n_elements, BLOCK_SIZE=1024)
        return NDArray(output)

    def cos(self):
        output = torch.empty_like(self.data, dtype=self.dtype)
        assert self.data.is_cuda and output.is_cuda
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        cos_kernel[grid](self.data, output, n_elements, BLOCK_SIZE=1024)
        return NDArray(output)

    def sin(self):
        output = torch.empty_like(self.data, dtype=self.dtype)
        assert self.data.is_cuda and output.is_cuda
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        sin_kernel[grid](self.data, output, n_elements, BLOCK_SIZE=1024)
        return NDArray(output)

    def sqrt(self):
        output = torch.empty_like(self.data, dtype=self.dtype)
        assert self.data.is_cuda and output.is_cuda
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        sqrt_kernel[grid](self.data, output, n_elements, BLOCK_SIZE=1024)
        return NDArray(output)

    def reshape(self, new_shape):
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        out_tensor = NDArray(self.data.reshape(new_shape))
        return out_tensor

    def permute(self, new_axis):
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        out_tensor = NDArray(self.data.permute(new_axis))
        if out_tensor.data.is_contiguous():
            out_tensor.data = out_tensor.data.contiguous()
        return out_tensor

    def broadcast_to(self, new_shape):
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        out_tensor = NDArray(self.data.broadcast_to(new_shape))
        if out_tensor.data.is_contiguous():
            out_tensor.data = out_tensor.data.contiguous()
        return out_tensor

    def __getitem__(self, idxs):
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        out_tensor = NDArray(self.data.__getitem__(idxs))
        return out_tensor

    def __setitem__(self, idxs, other):
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        if other.data.is_contiguous() is False:
            other.data = other.data.contiguous()
        self.data.__setitem__(idxs, other.data)
        return self

    def __eq__(self, other):
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        if isinstance(other, NDArray):
            if other.data.is_contiguous() is False:
                other.data = other.data.contiguous()
            out_tensor = NDArray(self.data.__eq__(other.data))
        else:
            out_tensor = NDArray(self.data.__eq__(other))
        return out_tensor

    def __ge__(self, other):
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        if isinstance(other, NDArray):
            if other.data.is_contiguous() is False:
                other.data = other.data.contiguous()
            out_tensor = NDArray(self.data.__ge__(other.data))
        else:
            out_tensor = NDArray(self.data.__ge__(other))
        return out_tensor

    def __ne__(self, other):
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        if isinstance(other, NDArray):
            if other.data.is_contiguous() is False:
                other.data = other.data.contiguous()
        return 1 - (self == other)

    def __gt__(self, other):
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        if isinstance(other, NDArray):
            if other.data.is_contiguous() is False:
                other.data = other.data.contiguous()
        return (self >= other) * (self != other)

    def __lt__(self, other):
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        if isinstance(other, NDArray):
            if other.data.is_contiguous() is False:
                other.data = other.data.contiguous()
        return 1 - (self >= other)

    def __le__(self, other):
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        if isinstance(other, NDArray):
            if other.data.is_contiguous() is False:
                other.data = other.data.contiguous()
        return 1 - (self > other)


    def __matmul__(self, b, activation=""):
        start_time = time.time()
        # Check constraints.
        assert self.shape[-1] == b.shape[-2], "Incompatible dimensions"
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        if b.data.is_contiguous() is False:
            b.data = b.data.contiguous()
        if len(self.shape) == 2:
            M, K = self.shape
            K, N = b.shape
            # Allocates output.
            c = torch.empty((M, N), device=torch.device("cuda"), dtype=self.dtype)
            # 1D launch kernel where each block gets its own program.
            grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
            matmul_kernel[grid](
                    self.data, b.data, c,
                    M, N, K,
                    self.stride(0), self.stride(1),
                    b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1),
                    BLOCK_SIZE_M=64,
                    BLOCK_SIZE_N=64,
                    BLOCK_SIZE_K=32,
                    GROUP_SIZE_M=8,
                    ACTIVATION=activation
            )
            return NDArray(c)
        elif len(self.shape) == 3 and len(self.shape) == 3:
            bz1, M, K = self.shape
            bz2, K, N = b.shape
            assert bz1 == bz2, "Batch sizes do not match!"
            c = torch.empty((bz1, M, N), device=self.data.device, dtype=self.dtype)
            # 1D launch kernel where each block gets its own program.
            grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
            for i in range(bz1):
                matmul_kernel[grid](
                        self.data[i], b.data[i], c.data[i],
                        M, N, K,
                        self.stride(-2), self.stride(-1),
                        b.stride(-2), b.stride(-1),
                        c.stride(-2), c.stride(-1),
                        BLOCK_SIZE_M=64,
                        BLOCK_SIZE_N=64,
                        BLOCK_SIZE_K=32,
                        GROUP_SIZE_M=8,
                        ACTIVATION=activation
                )
            return NDArray(c)


    def sum(self, axis=None, keepdims=False):
        shape = self.shape
        ndim = len(shape)
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        if axis is None:
            axis = tuple(range(ndim))
        elif isinstance(axis, int):
            axis = (axis,)
        else:
            axis = tuple(axis)

        # Create a permutation that moves all axes to be reduced to the end
        axes_to_keep = tuple(i for i in range(ndim) if i not in axis)
        new_order = axes_to_keep + axis

        self = self.permute(new_order)

        # Calculate the new shape after permutation
        new_shape = tuple(shape[i] for i in axes_to_keep) + tuple(shape[i] for i in axis)

        # Determine the dimensions for reduction
        m = prod(new_shape[:len(axes_to_keep)])
        n = prod(new_shape[len(axes_to_keep):])
        self = self.reshape((m, n))

        output_shape = tuple(new_shape[i] for i in range(len(axes_to_keep)))
        if keepdims:
            output_shape = list(shape)
            for i in axis:
                output_shape[i] = 1
            output_shape = tuple(output_shape)
        output = torch.empty(output_shape, device=torch.device("cuda"), dtype=self.dtype)

        block_m = 4
        block_n = triton.next_power_of_2(n)
        grid = (triton.cdiv(m, block_m), 1, 1)

        sum_kernel[grid](self.data, output, m, n, block_m, block_n)
        return NDArray(output)


    def max(self, axis=None, keepdims=False):
        shape = self.shape
        ndim = len(shape)
        if self.data.is_contiguous() is False:
            self.data = self.data.contiguous()
        if axis is None:
            axis = tuple(range(ndim))
        elif isinstance(axis, int):
            axis = (axis,)
        else:
            axis = tuple(axis)

        # Create a permutation that moves all axes to be reduced to the end
        axes_to_keep = tuple(i for i in range(ndim) if i not in axis)
        new_order = axes_to_keep + axis

        self = self.permute(new_order)

        # Calculate the new shape after permutation
        new_shape = tuple(shape[i] for i in axes_to_keep) + tuple(shape[i] for i in axis)

        # Determine the dimensions for reduction
        m = prod(new_shape[:len(axes_to_keep)])
        n = prod(new_shape[len(axes_to_keep):])
        self = self.reshape((m, n))

        output_shape = tuple(new_shape[i] for i in range(len(axes_to_keep)))
        if keepdims:
            output_shape = list(shape)
            for i in axis:
                output_shape[i] = 1
            output_shape = tuple(output_shape)
        output = torch.empty(output_shape, device=torch.device("cuda"), dtype=self.dtype)

        block_m = 4
        block_n = triton.next_power_of_2(n)
        grid = (triton.cdiv(m, block_m), 1, 1)

        max_kernel[grid](self.data, output, m, n, block_m, block_n)
        return NDArray(output)
