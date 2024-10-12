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

    def compute(self, *args) -> NDArray:
        """
        Calculate forward pass of operator.

        Args:
            input (NDArray): A list of input arrays to the function

        Returns:
            Array: Array output of the operation
        """
        raise NotImplementedError()

    def gradient(self, out_grad: "Value", node: "Value") -> Union["Value", Tuple["Value"]]:
        """
        Compute partial adjoint for each input value for a given output adjoint.

        Args:
            out_grad (Value): The adjoint with respect to the output value. 
            node (Value): The value node of forward evaluation.

        Returns:
            Value or Tuple[Value]: A list containing partial gradient adjoints to be propagated to each of the input node.
        """
        raise NotImplementedError()

    def gradient_as_tuple(self, out_grad: "Value", node: "Value") -> Tuple["Value"]:
        """
        Convenience method to always return a tuple from gradient call

        Args:
            out_grad (Value): The adjoint wrt to the output value
            node (Value): The Value node of forward evaluation

        Returns:
            Tuple[Value]: A tuple containing partial gradient adjoints to be propagated to each of the input node.
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

    def realize_cached_data(self) -> NDArray:
        """
        Run compute to realize the cached data
        """
        # avoid recomputation
        if self.cached_data is not None:
            return self.cached_data
        # note: data implicitly calls realized cached data
        self.cached_data = self.op.compute(*[x.realize_cached_data() for x in self.inputs])
        return self.cached_data

    def is_leaf(self) -> bool:
        """
        check current value is the leaf node in the computation graph
        """
        return self.op is None

    def init(
            self,
            op: Optional[Op],
            inputs: List["Tensor"],
            *,
            num_outputs: int = 1,
            cached_data: List[object] = None,
            requires_grad: Optional[bool] = None):
        """
        Initialize a new Tensor object with the given operation and input tensors.

        Args:
            op (Optional[Op]): The operation producing this tensor, if any. It can be None if the tensor is created directly without an operation.
            inputs (List["Tensor"]): A list of input Tensor objects that this tensor depends on.
            num_outputs (int, optional): The number of outputs the operation produces. Default is 1.
            cached_data (List[object], optional): Pre-computed data or intermediates that can be reused. None by default.
            requires_grad (Optional[bool], optional): Whether this tensor requires the computation of gradients. If None, it is inferred from the input tensors.
        """
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
    def make_const(cls, data: NDArray, *, requires_grad=False) -> "Value":
        """
        make const
        """
        value = cls.__new__(cls)
        value.init(None, [], cached_data=data, requires_grad=requires_grad)
        return value

    @classmethod
    def make_from_op(cls, op: Op, inputs: List["Value"]) -> "Value":
        """
        make from op
        """
        value = cls.__new__(cls)
        value.init(op, inputs)
        if not value.requires_grad:
            return value.detach()
        value.realize_cached_data()
        return value


class Tensor(Value):
    """
    basic type
    """
    grad: "Tensor"

    def __init__(self, array, *, device: Optional[Device] = None, dtype=None, requires_grad=True, **kwargs):
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

        self.init(None, [], cached_data=cached_data, requires_grad=requires_grad)

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
        tensor.init(op, inputs)
        tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        """
        make const
        """
        tensor = Tensor.__new__(Tensor)
        if isinstance(data, Tensor):
            tensor.init(None, [], cached_data=data.realize_cached_data(), requires_grad=requires_grad)
        else:
            tensor.init(None, [], cached_data=data, requires_grad=requires_grad)
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (value.dtype, self.dtype)
        self.cached_data = value.realize_cached_data()

    def copy_(self, value):
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

    def set_device(self):
        self.cached_data = array_api.array(self.cached_data, device=thanos.cuda())

    def __repr__(self):
        return "Id:" + str(id(self)) + " Tensor(" + str(self.realize_cached_data()) + ")"

    def __len__(self):
        return self.realize_cached_data().shape[0]

    def __getitem__(self, index):
        tensor = Tensor.__new__(Tensor)
        tensor.init(None, [], cached_data=self.realize_cached_data()[index], requires_grad=self.requires_grad)
        return tensor

    def __str__(self):
        return str(self.realize_cached_data())

    def numpy(self):
        data = self.realize_cached_data()
        if array_api is numpy:
            return data
        return data.numpy()

    def __add__(self, other):
        if isinstance(other, Tensor):
            return thanos.nn.functional.EWiseAdd()(self, other)
        else:
            return thanos.nn.functional.AddScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return thanos.nn.functional.EWiseAdd()(self, thanos.nn.functional.Negate()(other))
        else:
            return thanos.nn.functional.AddScalar(-other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return thanos.nn.functional.EWiseMul()(self, other)
        else:
            return thanos.nn.functional.MulScalar(other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return thanos.nn.functional.EWiseDiv()(self, other)
        else:
            return thanos.nn.functional.DivScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            raise TypeError("pow value must be a scalar")
        else:
            return thanos.nn.functional.PowScalar(other)(self)

    def __neg__(self):
        return thanos.nn.functional.Negate()(self)


    def equal(self, other):
        if isinstance(other, Tensor):
            return thanos.nn.functional.Equal()(self, other)
        else:
            return thanos.nn.functional.Equal()(self, other)

    def sin(self):
        return thanos.nn.functional.Sin()(self)

    def cos(self):
        return thanos.nn.functional.Cos()(self)

    def log(self):
        return thanos.nn.functional.Log()(self)

    def exp(self):
        return thanos.nn.functional.Exp()(self)

    def transpose(self, axis=None):
        return thanos.nn.functional.Transpose(axis)(self)

    def reshape(self, shape):
        return thanos.nn.functional.Reshape(shape)(self)

    def summation(self, axis=None, keepdims=False):
        return thanos.nn.functional.Summation(axis, keepdism=keepdims)(self)

    def sum(self, axis=None, keepdims=False):
        return thanos.nn.functional.Summation(axis, keepdims=keepdims)(self)

    def broadcast_to(self, shape):
        return thanos.nn.functional.BroadcastTo(shape)(self)

    def __matmul__(self, other):
        return thanos.nn.functional.Matmul()(self, other)

    def matmul(self, other):
        return thanos.nn.functional.Matmul()(self, other)

    def sqrt(self):
        return thanos.nn.functional.Sqrt()(self)


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
        return thanos.nn.functional.tuple_get_item(self, index)

    def tuple(self):
        return tuple([x for x in self])
    
    def __repr__(self):
        return "thanos.TensorTuple" + str(self.tuple())

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        return thanos.nn.functional.make_tuple(*[self[i] + other[i] for i in range(len(self))])

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
