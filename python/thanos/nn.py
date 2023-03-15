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

def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []

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

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

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

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.p > 0.0:
            mask = init.randb(*x.shape, p=(1 - self.p), dtype=x.dtype)
            x = x * mask / (1 - self.p)
        return x



class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor):
        return self.fn(x) + x

class BatchNorm1d(Module):
    """
    其中，通过均值和方差来计算标准化因子的公式如下：
    $$ \hat{x_i} = \frac{x_i - E[x]}{\sqrt{Var[x]+ \epsilon}} $$
    其中， $\epsilon$ 为防止除数为0的小常数， $E[x]$ 和 $Var[x]$ 分别为当前batch内 $x$ 的均值和方差。
    对输入数据进行归一化和平移的公式如下：
    $$ y_i = \gamma \hat{x_i} + \beta $$
    其中， $\gamma$ 和 $\beta$ 为可学习的参数，分别用于缩放和偏移标准化后的数据。
    """
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            batch = x.shape[0]
            mean = ops.summation(x, axes=0) / batch
            # remember to detach the mean
            self.running_mean = (self.momentum * mean.detach() + (1 - self.momentum) * self.running_mean).detach()
            mean = ops.broadcast_to(ops.reshape(mean, (1, self.dim)), x.shape)
            var = ops.summation((x - mean) ** 2, axes=0) / batch
            # remember to detach the var
            self.running_var = (self.momentum * var.detach() + (1 - self.momentum) * self.running_var).detach()
            var = ops.broadcast_to(ops.reshape(var, (1, self.dim)), x.shape).detach()
        else:
            mean = self.running_mean.reshape((1, self.dim)).broadcast_to(x.shape)
            var = self.running_var.reshape((1, self.dim)).broadcast_to(x.shape)
        x = (x - mean) / (var + self.eps) ** 0.5
        w = ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape)
        b = ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)
        x = w * x + b
        return x

class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        num, classes = logits.shape
        y_one_hot = init.one_hot(classes, y, dtype=logits.dtype)
        logsum = ops.logsumexp(logits, axes=(1,))
        logits_y = ops.summation(logits * y_one_hot, axes=(1,))
        loss = logsum - logits_y
        return ops.summation(loss) / logits.shape[0]
