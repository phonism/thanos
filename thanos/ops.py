"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
import numpy
from .autograd import TensorOp, NDArray, Tensor

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class Negate(TensorOp):
    def compute(self, a):
        return array_api.negative(a)

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * self.scalar


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWiseDiv(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a / b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / rhs / rhs


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a / self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class PowScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        hs, = node.inputs
        return self.scalar * out_grad * pow_scalar(hs, self.scalar - 1)


def pow_scalar(a, scalar):
    return PowScalar(scalar)(a)

class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        hs, = node.inputs
        return out_grad / hs

def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        hs, = node.inputs
        return out_grad * exp(hs)

def exp(a):
    return Exp()(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            return array_api.swapaxes(a, -1, -2)
        return array_api.swapaxes(a, self.axes[0], self.axes[1])

    def gradient(self, out_grad: Tensor, node: Tensor):
        if self.axes is None:
            return transpose(out_grad, (-1, -2))
        return transpose(out_grad, (self.axes[0], self.axes[1]))


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape: tuple):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        hs, = node.inputs
        return reshape(out_grad, hs.shape)

def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    """
    In order to broadcast, the size of the trailing axes for both arrays in an operation must either be the same size or one of them must be one.
    """
    def __init__(self, shape: tuple):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        hs, = node.inputs
        input_shape = list(hs.shape)
        base_shape = [1] * (len(self.shape) - len(input_shape)) + input_shape
        axes = []
        for i in range(len(input_shape)):
            if base_shape[i] != self.shape[i]:
                axes.append(i)
        out_grad = summation(out_grad, axes=tuple(axes))
        return reshape(out_grad, input_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        if isinstance(self.axes, int):
            self.axes = (self.axes,)

    def compute(self, a):
        return array_api.sum(a, self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor):
        hs, = node.inputs
        if self.axes is None:
            axes = hs.shape
        else:
            axes = self.axes

        grad_shape = list(out_grad.shape)
        new_axes = []
        for x in axes:
            if x >= 0:
                new_axes.append(x)
            else:
                new_axes.append(x + len(hs.shape))
        for x in sorted(new_axes):
            grad_shape.insert(x, 1)
        return broadcast_to(reshape(out_grad, grad_shape), hs.shape)

def summation(a, axes):
    return Summation(axes)(a)



class Matmul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        lhs_grad = matmul(out_grad, transpose(rhs, (-1, -2)))
        rhs_grad = matmul(transpose(lhs, (-1, -2)), out_grad)

        dim1 = len(lhs.shape)
        dim2 = len(rhs.shape)
        dim3 = len(out_grad.shape)

        # ???????????????shape????????????????????????????????????broadcast????????????????????????broadcase???sum??????
        if dim3 > dim1:
            lhs_grad = summation(lhs_grad, tuple(range(dim3 - dim1)))
        if dim3 > dim2:
            rhs_grad = summation(rhs_grad, tuple(range(dim3 - dim2)))
        return lhs_grad, rhs_grad

def matmul(a, b):
    return Matmul()(a, b)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad: Tensor, node: Tensor):
        hs, = node.inputs
        input_relu = relu(hs).numpy()
        return out_grad * Tensor(input_relu > 0)

def relu(a):
    return ReLU()(a)
