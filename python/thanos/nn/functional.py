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

class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)
    
    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)

class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros(*value.shape, device=out_grad.device))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)

def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad.detach(), out_grad.detach()


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad.detach()


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class Negate(TensorOp):
    def compute(self, a):
        return array_api.negative(a)

    def gradient(self, out_grad, node):
        return -out_grad.detach()


def negate(a):
    return Negate()(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return (out_grad * rhs).detach(), (out_grad * lhs).detach()


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar).detach()


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWiseDiv(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return array_api.divide(a, b, dtype=a.dtype)

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return (out_grad / rhs).detach(), (-out_grad * lhs / rhs / rhs).detach()


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return array_api.divide(a, self.scalar, dtype=a.dtype)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad / self.scalar).detach()


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class PowScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        hs, = node.inputs
        return (self.scalar * out_grad * pow_scalar(hs, self.scalar - 1)).detach()


def pow_scalar(a, scalar):
    return PowScalar(scalar)(a)

class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        hs, = node.inputs
        return (out_grad / hs).detach()

def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        hs, = node.inputs
        return (out_grad * exp(hs)).detach()

def exp(a):
    return Exp()(a)

class Sqrt(TensorOp):
    def compute(self, a):
        return array_api.sqrt(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        hs, = node.inputs
        return (out_grad / (2 * sqrt(hs))).detach()

def sqrt(a):
    return Sqrt()(a)

class Transpose(TensorOp):
    def __init__(self, axis: Optional[tuple] = None):
        self.axis = axis

    def compute(self, a):
        if self.axis is None:
            return array_api.swapaxes(a, -1, -2)
        return array_api.swapaxes(a, self.axis[0], self.axis[1])

    def gradient(self, out_grad: Tensor, node: Tensor):
        if self.axis is None:
            return transpose(out_grad, (-1, -2))
        return transpose(out_grad, (self.axis[0], self.axis[1])).detach()


def transpose(a, axis=None):
    return Transpose(axis)(a)


class Reshape(TensorOp):
    def __init__(self, shape: tuple):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        hs, = node.inputs
        return reshape(out_grad, hs.shape).detach()

def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    """
    In order to broadcast, the size of the trailing axis for both arrays 
    in an operation must either be the same size or one of them must be one.
    """
    def __init__(self, shape: tuple):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        hs, = node.inputs
        input_shape = list(hs.shape)
        base_shape = [1] * (len(self.shape) - len(input_shape)) + input_shape
        axis = []
        for i in range(len(base_shape)):
            if base_shape[i] != self.shape[i]:
                axis.append(i)
        out_grad = summation(out_grad, axis=tuple(axis))
        
        return reshape(out_grad, input_shape).detach()


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axis: Optional[tuple] = None, keepdims: Optional[bool] = False):
        self.axis = axis
        if isinstance(self.axis, int):
            self.axis = (self.axis,)
        self.keepdims = keepdims

    def compute(self, a):
        return array_api.sum(a, self.axis, keepdims=self.keepdims)

    def gradient(self, out_grad: Tensor, node: Tensor):
        hs, = node.inputs
        if self.axis is None:
            axis = hs.shape
        else:
            axis = self.axis
        grad_shape = list(out_grad.shape)
        new_axis = []
        for x in axis:
            if x >= 0:
                new_axis.append(x)
            else:
                new_axis.append(x + len(hs.shape))
        for x in sorted(new_axis):
            grad_shape.insert(x, 1)
        return broadcast_to(reshape(out_grad, grad_shape), hs.shape).detach()

def summation(a, axis=None, keepdims=False):
    return Summation(axis, keepdims)(a)


class Matmul(TensorOp):
    def compute(self, a, b):
        a_shape = a.shape
        b_shape = b.shape
        pre_shape_a = []
        pre_shape_b = []
        pre_a = 1
        pre_b = 1
        if len(a_shape) > 2 or len(b_shape) > 2:
            for i in range(len(a_shape) - 2):
                pre_shape_a.append(a_shape[i])
                pre_a *= a_shape[i]
            a = array_api.reshape(a, (pre_a, a_shape[-2], a_shape[-1]))
            for i in range(len(b_shape) - 2):
                pre_shape_b.append(b_shape[i])
                pre_b *= b_shape[i]
            b = array_api.reshape(b, (pre_b, b_shape[-2], b_shape[-1]))

            if pre_a == 1:
                a = array_api.broadcast_to(a, (b.shape[0], a.shape[1], a.shape[2]))
            if pre_b == 1:
                b = array_api.broadcast_to(b, (a.shape[0], b.shape[1], b.shape[2]))


        c = a @ b
        if len(a_shape) > 2 or len(b_shape) > 2:
            if pre_a >= pre_b:
                c = array_api.reshape(c, tuple(pre_shape_a) + (a_shape[-2], b_shape[-1]))
            else:
                c = array_api.reshape(c, tuple(pre_shape_b) + (a_shape[-2], b_shape[-1]))
        return c

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        lhs_grad = matmul(out_grad, transpose(rhs, (-1, -2)))
        rhs_grad = matmul(transpose(lhs, (-1, -2)), out_grad)

        dim1 = len(lhs.shape)
        dim2 = len(rhs.shape)
        dim3 = len(out_grad.shape)

        # 如果输出的shape比输入高，说明在前面做了broadcast，那么就要把这些broadcast给sum起来
        if dim3 > dim1:
            lhs_grad = summation(lhs_grad, tuple(range(dim3 - dim1)))
        if dim3 > dim2:
            rhs_grad = summation(rhs_grad, tuple(range(dim3 - dim2)))
        return lhs_grad.detach(), rhs_grad.detach()

def matmul(a, b):
    return Matmul()(a, b)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad: Tensor, node: Tensor):
        hs, = node.inputs
        input_relu = relu(hs).numpy()
        return (out_grad * Tensor(input_relu > 0, device=hs.device)).detach()

def relu(a):
    return ReLU()(a)

class LogSumExp(TensorOp):
    def __init__(self, axis: Optional[tuple] = None):
        self.axis = axis

    def compute(self, Z):
        self.max_value = Z.max(self.axis, keepdims=True)
        max_z = array_api.broadcast_to(self.max_value, Z.shape)
        Z = array_api.exp(Z - max_z)
        Z = array_api.sum(Z, self.axis)
        Z = array_api.log(Z)
        return Z + array_api.reshape(self.max_value, Z.shape)

    def gradient(self, out_grad, node):
        hs, = node.inputs
        input_shape = hs.shape
        max_z = array_api.broadcast_to(self.max_value, input_shape)
        base_shape = list(input_shape)
        if isinstance(self.axis, int): 
            self.axis = (self.axis,)
        axis = list(range(len(base_shape))) if self.axis is None else self.axis
        for ax in axis:
            base_shape[ax] = 1
        out_grad = out_grad / summation(exp(hs - max_z), self.axis)
        out_grad = reshape(out_grad, base_shape)
        out_grad = broadcast_to(out_grad, input_shape)
        out_grad = out_grad * exp(hs - max_z)
        return (out_grad.detach(), )

def logsumexp(a, axis=None):
    return LogSumExp(axis=axis)(a)

class Equal(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a == b

    def gradient(self, out_grad: Tensor, node: Tensor):
        a, b = node.inputs
        grad_a = array_api.reduce_sum(out_grad, axis=None, keepdims=False)
        grad_b = array_api.reduce_sum(out_grad, axis=None, keepdims=False)
        return grad_a.detach(), grad_b.detach()

def equal(a, b):
    return Equal()(a, b)


class Max(TensorOp):
    def __init__(self, axis: Optional[tuple] = None, keepdims: Optional[bool] =False):
        self.axis = axis
        if isinstance(self.axis, int):
            self.axis = (self.axis,)
        self.keepdims = keepdims

    def compute(self, a):
        return array_api.max(a, self.axis, keepdims=self.keepdims)

    def gradient(self, out_grad: Tensor, node: Tensor):
        # Your code here
        hs, = node.inputs
        if self.axis is None:
            axis = hs.shape
        else:
            axis = self.axis
        grad_shape = list(out_grad.shape)
        new_axis = []
        for x in axis:
            if x >= 0:
                new_axis.append(x)
            else:
                new_axis.append(x + len(hs.shape))
        for x in sorted(new_axis):
            grad_shape.insert(x, 1)
        mask = hs.equal(broadcast_to(max(hs, axis=self.axis, keepdims=True), hs.shape))
        return (broadcast_to(reshape(out_grad, grad_shape), hs.shape) * mask).detach()

def max(a, axis=None, keepdims=False):
    return Max(axis, keepdims)(a)

class Stack(TensorOp):
    def __init__(self, axis: int):
        self.axis = axis

    def compute(self, tensors):
        in_shape = tensors[0].shape
        out_shape = [len(tensors)] + list(in_shape)
        out = NDArray.make(out_shape, device=tensors[0].device)
        idxs = [slice(None, None, None) for j in range(len(in_shape))]
        for i, arg in enumerate(tensors):
            assert arg.shape == in_shape
            idxs_i = tuple([i] + idxs)
            out[idxs_i] = arg.compact()
        out_axes = list(range(1, len(out_shape)))
        out_axes.insert(self.axis, 0)
        return out.permute(tuple(out_axes)).compact()
    
    def gradient(self, out_grad, node):
        return split(out_grad, self.axis)

def stack(tensors, axis=0):
    return Stack(axis)(make_tuple(*tensors))

class Split(TensorTupleOp):
    def __init__(self, axis: int):
        self.axis = axis

    def compute(self, x):
        if self.axis < 0:
            self.axis = self.axis + len(x.shape)
        in_shape = x.shape
        idx = [slice(None, None, None) for j in range(len(in_shape))]
        results = []
        for i in range(in_shape[self.axis]):
            idx_i = idx.copy()
            idx_i[self.axis] = i
            idx_i = tuple(idx_i)
            out = x[idx_i]
            out = out.sum(axis=self.axis)
            results.append(out)
        return tuple(results)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return stack(out_grad, self.axis)

def split(a, axis):
    return Split(axis)(a)
