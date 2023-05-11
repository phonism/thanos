"""
The module
"""

from typing import List, Callable, Any
import thanos
from ..autograd import Tensor
import thanos.nn.functional as F
from thanos import init
import thanos.backend_ndarray as nd
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

def _unpack_vars(value: object) -> List[Tensor]:
    if isinstance(value, Tensor):
        return [value]
    elif isinstance(value, Module):
        return value.vars()
    elif isinstance(value, dict):
        var_list = []
        for k, v in value.items():
            var_list += _unpack_vars(v)
        return var_list
    elif isinstance(value, (list, tuple)):
        var_list = []
        for v in value:
            var_list += _unpack_vars(v)
        return var_list
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

    def vars(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_vars(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def train(self):
        self.training = True
        for m in self._children():
            m.traning = True

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def cuda(self):
        for idx in range(len(self.parameters())):
            self.parameters()[idx].set_device()
        for idx in range(len(self.vars())):
            self.vars()[idx].set_device()
        for idx in range(len(self._children())):
            self._children()[idx].cuda()

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
                    F.transpose(init.kaiming_uniform(self.out_features, 1)),
                    device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = F.matmul(x, self.weight)
        if self.bias:
            x = x + F.broadcast_to(F.reshape(self.bias, (1,) * (len(x.shape) - 1) + (self.out_features,)), x.shape)
        return x


class Flatten(Module):
    def forward(self, x) -> Tensor:
        return F.reshape(x, (x.shape[0], -1))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(x)
        return x


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.p > 0.0:
            mask = init.randb(*x.shape, p=(1 - self.p), dtype=x.dtype, device=x.device)
            x = x * mask / (1 - self.p)
        return x

class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x

class BatchNorm1d(Module):
    def __init__(self, dim: int, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
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
            mean = F.summation(x, axis=0) / batch
            # remember to detach the mean
            self.running_mean = (self.momentum * mean.detach() + (1 - self.momentum) * self.running_mean).detach()
            mean = F.broadcast_to(F.reshape(mean, (1, self.dim)), x.shape)
            var = F.summation((x - mean) ** 2, axis=0) / batch
            # remember to detach the var
            self.running_var = (self.momentum * var.detach() + (1 - self.momentum) * self.running_var).detach()
            var = F.broadcast_to(F.reshape(var, (1, self.dim)), x.shape)
        else:
            mean = self.running_mean.reshape((1, self.dim)).broadcast_to(x.shape)
            var = self.running_var.reshape((1, self.dim)).broadcast_to(x.shape)
        x = (x - mean) / (var + self.eps) ** 0.5
        w = F.broadcast_to(F.reshape(self.weight, (1, self.dim)), x.shape)
        b = F.broadcast_to(F.reshape(self.bias, (1, self.dim)), x.shape)
        x = w * x + b
        return x

class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.beta = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x):
        if x.shape[-1] != self.dim:
            raise RuntimeError('Input dims should be %d' % self.dim)
        mean = F.summation(x, axis=-1) / x.shape[-1]
        mean = F.broadcast_to(F.reshape(mean, mean.shape + (1,)), x.shape)
        var = F.summation((x - mean) ** 2, axis=-1) / self.dim
        var = F.broadcast_to(F.reshape(var, var.shape + (1,)), x.shape)
        gamma = F.broadcast_to(F.reshape(self.gamma, (1, ) * (len(x.shape) - 1) + (self.dim,)), x.shape)
        beta = F.broadcast_to(F.reshape(self.beta, (1, ) * (len(x.shape) - 1) + (self.dim,)), x.shape)
        output = (x - mean) / F.sqrt(var + self.eps)
        output = gamma * output + beta 
        return output

class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = Parameter(init.ones(dim))

    def forward(self, x):
        rms = x / F.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return rms * self.weight

class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        num, classes = logits.shape
        y_one_hot = init.one_hot(classes, y, dtype=logits.dtype, device=logits.device)
        logsum = F.logsumexp(logits, axis=(1,))
        logits_y = F.summation(logits * y_one_hot, axis=(1,))
        loss = logsum - logits_y
        return F.summation(loss) / logits.shape[0]

class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        x_exp = F.exp(x - F.broadcast_to(F.max(x, self.dim, keepdims=True), x.shape))
        x = x_exp / F.broadcast_to(F.summation(x_exp, axis=self.dim, keepdims=True), x.shape)
        return x

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim))

    def forward(self, x):
        x_one_hot = init.one_hot(self.num_embeddings, x.realize_cached_data().flat, device=x.device)
        res = x_one_hot @ self.weight
        return res.reshape((*x.shape, self.embedding_dim))

class SiLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / (F.exp(-x) + 1)


class MultiheadAttention(Module):
    def __init__(self, dim=64, heads=1, device=None, dtype="float32"):
        self.dim = dim
        self.heads = heads
        self.w_kqv = Parameter(
                init.kaiming_uniform(self.dim, self.dim * 3),
                device=device, dtype=dtype)
        self.w_out = Parameter(
                init.kaiming_uniform(self.dim, self.dim),
                device=device, dtype=dtype)
        self.softmax = Softmax()

    def forward(self, x: Tensor) -> Tensor:
        k, q, v = F.split(F.reshape(x @ self.w_kqv, (x.shape[0], x.shape[1], 3, self.dim)), axis=2)
        k, q, v = [F.reshape(a, (x.shape[0], x.shape[1], self.heads, self.dim // self.heads)).transpose((1, 2)) for a in [k, q, v]]
        mask = thanos.triu((-float("inf") * init.ones(x.shape[1], x.shape[1], device=x.device)), k=1)
        mask = F.broadcast_to(F.reshape(mask, (1, 1,) + mask.shape), (k.shape[0], k.shape[1],) + mask.shape)
        atten = self.softmax(k @ F.transpose(q) / np.sqrt(self.dim // self.heads) + mask)
        return F.reshape((atten @ v).transpose((1, 2)), (x.shape[0], x.shape[1], self.dim)) @ self.w_out, atten
