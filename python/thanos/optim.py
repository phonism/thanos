"""Optimization module"""
import thanos
import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None

class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    
    def step(self):
        for idx, p in enumerate(self.params):
            grad = p.grad.detach() + self.weight_decay * p.detach()
            if idx not in self.u.keys():
                self.u[idx] = 0
            self.u[idx] = self.momentum * self.u[idx] + (1 - self.momentum) * grad
            self.params[idx].data = p.data - self.lr * self.u[idx]
