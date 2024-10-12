"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
import numpy
from ..autograd import TensorOp, NDArray, Tensor, TensorTuple, TensorTupleOp, Value
from thanos import init

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
#import numpy as array_api
from ..backend_selection import array_api, NDArray

import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_kernel(X, Y, W, B, Mean, Rstd, stride, N, eps, BLOCK_SIZE: tl.constexpr,):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)

@triton.jit
def _layer_norm_bwd_dx_kernel(DX, DY, DW, DB, X, W, Mean, Rstd, Lock, stride, 
        N, GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    # Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols
    # Load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    # Compute dx
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    # Write dx
    tl.store(DX + cols, dx, mask=mask)
    # Accumulate partial sums for dw/db
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # First store doesn't accumulate
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)
    # Release the lock
    tl.atomic_xchg(Lock, 0)

@triton.jit
def _layer_norm_bwd_dwdb_kernel(DW, DB, FINAL_DW, FINAL_DB, M, N, 
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of DW and DB it should compute.
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Iterate through the rows of DW and DB to sum the partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)

class FusedLayerNormFunction(TensorOp):
    def __init__(self, eps=1e-6):
        self.eps = eps

    def compute(self, x, weight, bias):
        # allocate output
        x = x.data
        weight = weight.data
        bias = bias.data
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape((-1, x.shape[-1]))
        M, N = x_arg.shape
        self.mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
        self.rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = 8
        # enqueue kernel
        _layer_norm_fwd_kernel[(M, )](
                x_arg, y, weight, bias, self.mean, self.rstd, 
                x_arg.stride(0), N, self.eps, 
                BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
        self.BLOCK_SIZE = BLOCK_SIZE
        self.num_warps = num_warps
        return NDArray(y)

    def gradient(self, out_grad: Tensor, node: Tensor):
        x, w, b = node.inputs
        x = x.realize_cached_data().data
        w = w.realize_cached_data().data
        b = b.realize_cached_data().data
        m = self.mean
        v = self.rstd
        #x, w, b, m, v = ctx.saved_tensors
        # heuristics for amount of parallel reduction stream for DW/DB
        N = w.shape[0]
        GROUP_SIZE_M = 64
        if N <= 8192: 
            GROUP_SIZE_M = 96
        if N <= 4096: 
            GROUP_SIZE_M = 128
        if N <= 1024: 
            GROUP_SIZE_M = 256
        # allocate output
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device)
        _dw = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        _db = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        dw = torch.empty((N, ), dtype=w.dtype, device=w.device)
        db = torch.empty((N, ), dtype=w.dtype, device=w.device)
        dy = out_grad.realize_cached_data().data
        dx = torch.empty_like(dy)
        # enqueue kernel using forward pass heuristics
        # also compute partial sums for DW and DB
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        # TODO!
        if dy.is_contiguous() is False:
            dy = dy.contiguous()
        _layer_norm_bwd_dx_kernel[(M, )](
                dx, dy, _dw, _db, x, w, m, v, locks,
                x_arg.stride(0), N, 
                BLOCK_SIZE_N=self.BLOCK_SIZE,
                GROUP_SIZE_M=GROUP_SIZE_M,
                num_warps=self.num_warps)
        grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
        # accumulate partial sums in separate kernel
        _layer_norm_bwd_dwdb_kernel[grid](
                _dw, _db, dw, db, min(GROUP_SIZE_M, M), N,
                BLOCK_SIZE_M=32,
                BLOCK_SIZE_N=128, num_ctas=1)
        return (Tensor(dx), Tensor(dw), Tensor(db))

def fused_layer_norm(x, weight, bias, eps=1e-6):
    return FusedLayerNormFunction(eps)(x, weight, bias)

@triton.jit
def _rms_norm_fwd_kernel(X, Y, W, Mean, Rstd, stride, N, eps, BLOCK_SIZE: tl.constexpr,):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a * a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    rstd = 1 / tl.sqrt(mean + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = x * rstd
        y = x_hat * w
        # Write output
        tl.store(Y + cols, y, mask=mask)

@triton.jit
def _rms_norm_bwd_dx_kernel(DX, DY, DW, X, W, Mean, Rstd, Lock, stride, 
        N, GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    # Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    # Load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    # Compute dx
    xhat = x * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd

    # Write dx
    tl.store(DX + cols, dx, mask=mask)
    # Accumulate partial sums for dw/db
    partial_dw = (dy * xhat).to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # First store doesn't accumulate
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    # Release the lock
    tl.atomic_xchg(Lock, 0)

@triton.jit
def _rms_norm_bwd_dw_kernel(DW, FINAL_DW, M, N, 
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of DW and DB it should compute.
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Iterate through the rows of DW and DB to sum the partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)

class FusedRMSNormFunction(TensorOp):
    def __init__(self, eps=1e-6):
        self.eps = eps

    def compute(self, x, weight):
        # allocate output
        x = x.data
        weight = weight.data
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape((-1, x.shape[-1]))
        M, N = x_arg.shape
        self.mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
        self.rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = 8
        # enqueue kernel
        _rms_norm_fwd_kernel[(M, )](
                x_arg, y, weight, self.mean, self.rstd, 
                x_arg.stride(0), N, self.eps, 
                BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
        self.BLOCK_SIZE = BLOCK_SIZE
        self.num_warps = num_warps
        return NDArray(y)

    def gradient(self, out_grad: Tensor, node: Tensor):
        x, w = node.inputs
        x = x.realize_cached_data().data
        w = w.realize_cached_data().data
        if x.is_contiguous() is False:
            x = x.contiguous()
        if w.is_contiguous() is False:
            w = w.contiguous()
        m = self.mean
        v = self.rstd
        #x, w, b, m, v = ctx.saved_tensors
        # heuristics for amount of parallel reduction stream for DW/DB
        N = w.shape[0]
        GROUP_SIZE_M = 64
        if N <= 8192: 
            GROUP_SIZE_M = 96
        if N <= 4096: 
            GROUP_SIZE_M = 128
        if N <= 1024: 
            GROUP_SIZE_M = 256
        # allocate output
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device)
        _dw = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        dw = torch.empty((N, ), dtype=w.dtype, device=w.device)
        dy = out_grad.realize_cached_data().data
        dx = torch.empty_like(dy)
        # enqueue kernel using forward pass heuristics
        # also compute partial sums for DW and DB
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        # TODO!
        if dy.is_contiguous() is False:
            dy = dy.contiguous()
        _rms_norm_bwd_dx_kernel[(M, )](
                dx, dy, _dw, x, w, m, v, locks,
                x_arg.stride(0), N, 
                BLOCK_SIZE_N=self.BLOCK_SIZE,
                GROUP_SIZE_M=GROUP_SIZE_M,
                num_warps=self.num_warps)
        grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
        # accumulate partial sums in separate kernel
        _rms_norm_bwd_dw_kernel[grid](
                _dw, dw, min(GROUP_SIZE_M, M), N,
                BLOCK_SIZE_M=32,
                BLOCK_SIZE_N=128, num_ctas=1)
        return (Tensor(dx), Tensor(dw))

def fused_rms_norm(x, weight, eps=1e-6):
    return FusedRMSNormFunction(eps)(x, weight)
