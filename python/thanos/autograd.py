import thanos
from typing import List, Optional, NamedTuple, Tuple, Union

import numpy
from thanos import init
#import numpy as array_api
#NDArray = numpy.ndarray
from .backend_selection import Device, array_api, NDArray, default_device

TENSOR_COUNTER = 0


class Op:
    """
    operator definitions
    """
    def __call__(self, *args):
        raise NotImplementedError()

    def compute(self, *args: Tuple[NDArray]) -> NDArray:
        """Calculate forward pass of operator.

        Parameters
        ----------
        input: NDArray
            A list of input arrays to the function

        Returns
        -------
        output: Array
            Array output of the operation
        """
        raise NotImplementedError()

    def gradient(self, out_grad: "Value", node: "Value") -> Union["Value", Tuple["Value"]]:
        """Compute partial adjoint for each input value for a given output adjoint.

        Parameters
        ----------
        out_grad: Value
            The adjoint wrt to the output value.

        node: Value
            The value node of forward evaluation.

        Returns
        -------
        input_grads: Value or Tuple[Value]
            A list containing partial gradient adjoints to be propagated to each of the input node.
        """
        raise NotImplementedError()

    def gradient_as_tuple(self, out_grad: "Value", node: "Value") -> Tuple["Value"]:
        """Convenience method to always return a tuple from gradient call

        Parameters
        ----------
        out_grad: Value
            The adjoint wrt to the output value

        node: Value
            The Value node of forward evaluation

        Returns
        -------
        input_grads: Tuple[Value]
            A tuple containing partial gradient adjoints to be propagated to each of the input node.
        """
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output, )


class Value:
    """
    A value in the computation graph
    """

    # trace of computational graph
    op: Optional[Op]
    inputs: List["Value"]
    # The following fields are cached fields for dynamic computation
    cached_data: NDArray
    requires_grad: bool

    def realize_cached_data(self):
        """
        Run compute to realize the cached data
        """
        # avoid recomputation
        if self.cached_data is not None:
            return self.cached_data
        # note: data implicitly calls realized cached data
        self.cached_data = self.op.compute(*[x.realize_cached_data() for x in self.inputs])
        return self.cached_data

    def is_leaf(self):
        """
        check current value is the leaf node in the computation graph
        """
        return self.op is None

    def _init(
            self,
            op: Optional[Op],
            inputs: List["Tensor"],
            *,
            num_outputs: int = 1,
            cached_data: List[object] = None,
            requires_grad: Optional[bool] = None):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1

        if requires_grad is None:
            # check the inputs op requires grad
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1


    def detach(self):
        """
        generate a const Value
        """
        return Value.make_const(self.realize_cached_data())

    @classmethod
    def make_const(cls, data, *, requires_grad=False):
        """
        make const
        """
        value = cls.__new__(cls)
        value._init(None, [], cached_data=data, requires_grad=requires_grad)
        return value

    @classmethod
    def make_from_op(cls, op: Op, inputs: List["Value"]):
        """
        make from op
        """
        value = cls.__new__(cls)
        value._init(op, inputs)
        if not value.requires_grad:
            return value.detach()
        value.realize_cached_data()
        return value


class Tensor(Value):
    """
    basic type
    """
    grad: "Tensor"

    def __init__(
            self,
            array,
            *,
            device: Optional[Device] = None,
            dtype=None,
            requires_grad=True,
            **kwargs):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                cached_data = Tensor._array_from_numpy(array.numpy(), device=device, dtype=dtype)
        else:
            device = device if device else default_device()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(None, [], cached_data=cached_data, requires_grad=requires_grad)

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        if array_api is numpy:
            return numpy.array(numpy_array, dtype=dtype)
        return array_api.array(numpy_array, device=device, dtype=dtype)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        """
        make from op
        """
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        """
        make const
        """
        tensor = Tensor.__new__(Tensor)
        if isinstance(data, Tensor):
            tensor._init(None, [], cached_data=data.realize_cached_data(), requires_grad=requires_grad)
        else:
            tensor._init(None, [], cached_data=data, requires_grad=requires_grad)
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (value.dtype, self.dtype)
        self.cached_data = value.realize_cached_data()

    def backward(self, out_grad=None):
        out_grad = out_grad if out_grad else init.ones(*self.shape, dtype=self.dtype, device=self.device)
        compute_gradient_of_variables(self, out_grad)

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    @property
    def device(self):
        data = self.realize_cached_data()
        # numpy array always sits on cpu
        if array_api is numpy:
            return default_device()
        return data.device

    def __repr__(self):
        return "Id:" + str(id(self)) + " Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return str(self.realize_cached_data())

    def numpy(self):
        data = self.realize_cached_data()
        if array_api is numpy:
            return data
        return data.numpy()

    def __add__(self, other):
        if isinstance(other, Tensor):
            return thanos.ops.EWiseAdd()(self, other)
        else:
            return thanos.ops.AddScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return thanos.ops.EWiseAdd()(self, thanos.ops.Negate()(other))
        else:
            return thanos.ops.AddScalar(-other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return thanos.ops.EWiseMul()(self, other)
        else:
            return thanos.ops.MulScalar(other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return thanos.ops.EWiseDiv()(self, other)
        else:
            return thanos.ops.DivScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            raise TypeError("pow value must be a scalar")
        else:
            return thanos.ops.PowScalar(other)(self)

    def __neg__(self):
        return thanos.ops.Negate()(self)

    def equal(self, other):
        if isinstance(other, Tensor):
            return thanos.ops.Equal()(self, other)
        else:
            return thanos.ops.Equal()(self, other)

    def log(self):
        return thanos.ops.Log()(self)

    def exp(self):
        return thanos.ops.Exp()(self)

    def transpose(self, axis=None):
        return thanos.ops.Transpose(axis)(self)

    def reshape(self, shape):
        return thanos.ops.Reshape(shape)(self)

    def summation(self, axis=None):
        return thanos.ops.Summation(axis)(self)

    def sum(self, axis=None):
        return thanos.ops.Summation(axis)(self)

    def broadcast_to(self, shape):
        return thanos.ops.BroadcastTo(shape)(self)

    def __matmul__(self, other):
        return thanos.ops.Matmul()(self, other)

    def matmul(self, other):
        return thanos.ops.Matmul()(self, other)


    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rmatmul__ = __matmul__

class TensorOp(Op):
    """ Op class specialized to output tensors, will be alterate subclasses for other structures """

    def __call__(self, *args):
        return Tensor.make_from_op(self, args)

class TensorTuple(Value):
    def __len__(self):
        cdata = self.realize_cached_data()
        return len(cdata)

    def __getitem__(self, index: int):
        return thanos.ops.tuple_get_item(self, index)

    def tuple(self):
        return tuple([x for x in self])
    
    def __repr__(self):
        return "thanos.TensorTuple" + str(self.tuple())

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        return thanos.ops.make_tuple(*[self[i] + other[i] for i in range(len(self))])

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return TensorTuple.make_const(self.realize_cached_data())

class TensorTupleOp(Op):
    def __call__(self, *args):
        return TensorTuple.make_from_op(self, args)


def compute_gradient_of_variables(out_tensor, out_grad):
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    node_to_output_grads_list[out_tensor] = [out_grad]
    reverse_topo_order = list(reversed(find_topo_sort([out_tensor])))

    for node in reverse_topo_order:
        node.grad = sum_node_list(node_to_output_grads_list[node])
        for nd in node.inputs:
            if nd not in node_to_output_grads_list:
                node_to_output_grads_list[nd] = []
        # 对于非叶子节点，将梯度传导到inputs上
        if not node.is_leaf():
            grad = node.op.gradient_as_tuple(node.grad, node)
            for i, nd in enumerate(node.inputs):
                node_to_output_grads_list[nd].append(grad[i])


def find_topo_sort(node_list: List[Value]) -> List[Value]:
    visited = dict()
    topo_order = list()
    for node in node_list:
        if node not in visited or visited[node] == 0:
            topo_sort_dfs(node, visited, topo_order)
    return topo_order

def topo_sort_dfs(node, visited, topo_order):
    if node in visited and visited[node] == 2:
        return
    visited[node] = 1
    for nd in node.inputs:
        if nd not in visited or visited[nd] == 0:
            topo_sort_dfs(nd, visited, topo_order)
    visited[node] = 2
    topo_order.append(node)

def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)
