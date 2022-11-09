"""
The module
"""

from typing import List, Callable, Any
import thanos
from thanos.autograd import Tensor
from thanos import ops
from thanos import init
import numpy as np

class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""

def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    return []


class Module:
    def __init__(self):
        self.training = True

    def _children(self):
        return _child_modules(self.__dict__)

    def train(self):
        self.training = True
        for m in self._children():
            m.traning = True

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
                init.kaiming_uniform(self.in_features, self.out_features),
                device=device, dtype=dtype)
        if bias:
            self.bias = Parameter(
                    ops.transpose(init.kaiming_uniform(self.out_features, 1)),
                    device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = ops.matmul(x, self.weight)
        if self.bias:
            x = x + ops.broadcast_to(self.bias, x.shape)
        return x


class Flatten(Module):
    def forward(self, x):
        return ops.reshape(x, (x.shape[0], -1))


class ReLU(Module):
    def forward(self, x: Tensor):
        x = ops.relu(x)
        return x


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
