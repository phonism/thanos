import sys
sys.path.append('./python')
import itertools
import numpy as np
import pytest
import torch

import thanos
import thanos.nn.functional as F
#from thanos import backend_ndarray as nd

def backward_check(f, *args, **kwargs):
    eps = 1e-5
    out = f(*args, **kwargs)
    c = np.random.randn(*out.shape)
    numerical_grad = [np.zeros(a.shape) for a in args]
    num_args = len(args)
    for i in range(num_args):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] += eps
            numerical_grad[i].flat[j] = (f1 - f2) / (2 * eps)
    backward_grad = out.op.gradient_as_tuple(thanos.Tensor(c, device=args[0].device), out)
    error = sum(
            np.linalg.norm(backward_grad[i].numpy() - numerical_grad[i]) for i in range(len(args)))
    assert error < 4.2e-1
    return [g.numpy() for g in backward_grad]

_DEVICES = [
    thanos.cpu(), 
    pytest.param(
            thanos.cuda(), 
            marks=pytest.mark.skipif(not thanos.cuda().enabled(), reason="No GPU"))]

EWISE_OPS = {
    "add": lambda a, b: a + b,
    "divide": lambda a, b: a / b,
    "subtract": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
}
EWISE_OP_FNS = [EWISE_OPS[k] for k in EWISE_OPS]
EWISE_OP_NAMES = [k for k in EWISE_OPS]
GENERAL_SHAPES = [(1, 1, 1), (4, 5, 6)]
@pytest.mark.parametrize("fn", EWISE_OP_FNS, ids=EWISE_OP_NAMES)
@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ewise_fn(fn, shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    _B = np.random.randn(*shape).astype(np.float32)
    A = thanos.Tensor(_A, device=device)
    B = thanos.Tensor(_B, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    TB = torch.Tensor(_B)
    TB.requires_grad = True
    np.testing.assert_allclose(fn(TA, TB).detach().numpy(), fn(A, B).detach().numpy(), atol=1e-5, rtol=1e-5)

    fn(TA, TB).sum().backward()
    fn(A, B).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(TB.grad.numpy(), B.grad.numpy(), atol=1e-5, rtol=1e-5)


SCALAR_OPS = {
    "add": lambda a, b: a + b,
    "divide": lambda a, b: a / b,
    "subtract": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "power": lambda a, b: a ** b,
}
SCALAR_OP_FNS = [SCALAR_OPS[k] for k in SCALAR_OPS]
SCALAR_OP_NAMES = [k for k in SCALAR_OPS]
GENERAL_SHAPES = [(1, 1, 1), (4, 5, 6)]
@pytest.mark.parametrize("fn", SCALAR_OP_FNS, ids=SCALAR_OP_NAMES)
@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_scalar_fn(fn, shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    _B = np.random.randn(1).astype(np.float32).item()
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    A = thanos.Tensor(_A, device=device)
    np.testing.assert_allclose(fn(TA, _B).detach().numpy(), fn(A, _B).detach().numpy(), atol=1e-5, rtol=1e-5)

    fn(TA, _B).sum().backward()
    fn(A, _B).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)


MATMUL_DIMS = [
    (16, 16, 16),
    (8, 8, 8),
    (1, 2, 3),
    (3, 4, 5),
    (5, 4, 3),
    (16, 16, 32),
    (64, 64, 64),
    (72, 72, 72),
    (72, 73, 74),
    (74, 73, 72),
    (128, 128, 128)]
@pytest.mark.parametrize("m,n,p", MATMUL_DIMS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_matmul(m, n, p, device):
    _A = np.random.randn(m, n).astype(np.float32)
    _B = np.random.randn(n, p).astype(np.float32)
    A = thanos.Tensor(_A, device=device)
    B = thanos.Tensor(_B, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad=True
    TB = torch.Tensor(_B)
    TB.requires_grad=True
    np.testing.assert_allclose((TA @ TB).detach().numpy(), (A @ B).detach().numpy(), atol=1e-5, rtol=1e-5)

    (TA @ TB).sum().backward()
    (A @ B).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)

BATCH_MATMUL_DIMS = [
    (16, 16, 16, 16),
    (32, 16, 8, 24),
    (32, 13, 8, 15),
]
@pytest.mark.parametrize("b,m,n,p", BATCH_MATMUL_DIMS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_batch_matmul(b, m, n, p, device):
    _A = np.random.randn(b, m, n).astype(np.float32)
    _B = np.random.randn(b, n, p).astype(np.float32)
    A = thanos.Tensor(_A, device=device)
    B = thanos.Tensor(_B, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad=True
    TB = torch.Tensor(_B)
    TB.requires_grad=True
    np.testing.assert_allclose((TA @ TB).detach().numpy(), (A @ B).detach().numpy(), atol=1e-5, rtol=1e-5)

    (TA @ TB).sum().backward()
    (A @ B).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("b,m,n,p", BATCH_MATMUL_DIMS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_batch_matmul_2(b, m, n, p, device):
    _A = np.random.randn(b, m, n).astype(np.float32)
    _B = np.random.randn(n, p).astype(np.float32)
    A = thanos.Tensor(_A, device=device)
    B = thanos.Tensor(_B, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad=True
    TB = torch.Tensor(_B)
    TB.requires_grad=True
    np.testing.assert_allclose((TA @ TB).detach().numpy(), (A @ B).detach().numpy(), atol=1e-5, rtol=1e-5)

    (TA @ TB).sum().backward()
    (A @ B).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("b,m,n,p", BATCH_MATMUL_DIMS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_batch_matmul_3(b, m, n, p, device):
    _A = np.random.randn(m, n).astype(np.float32)
    _B = np.random.randn(b, n, p).astype(np.float32)
    A = thanos.Tensor(_A, device=device)
    B = thanos.Tensor(_B, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad=True
    TB = torch.Tensor(_B)
    TB.requires_grad=True
    np.testing.assert_allclose((TA @ TB).detach().numpy(), (A @ B).detach().numpy(), atol=1e-5, rtol=1e-5)

    (TA @ TB).sum().backward()
    (A @ B).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)

SUMMATION_PARAMETERS = [
    ((1, 1, 1), None),
    ((5, 3), 0),
    ((8, 3, 2), 1),
    ((8, 3, 2), 2)
]
@pytest.mark.parametrize("shape, axes", SUMMATION_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_summation(shape, axes, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = thanos.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    np.testing.assert_allclose(
            torch.sum(TA, dim=axes).detach().numpy(), 
            F.summation(A, axis=axes).detach().numpy(), atol=1e-5, rtol=1e-5)

    torch.sum(TA, dim=axes).sum().backward()
    F.summation(A, axis=axes).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_log(shape, device):
    _A = np.random.randn(*shape).astype(np.float32) + 5.
    A = thanos.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    np.testing.assert_allclose(
            torch.log(TA).detach().numpy(), 
            F.log(A).detach().numpy(), atol=1e-5, rtol=1e-5)

    torch.log(TA).sum().backward()
    F.log(A).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_exp(shape, device):
    _A = np.random.randn(*shape).astype(np.float32) + 5.
    A = thanos.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    np.testing.assert_allclose(
            torch.exp(TA).detach().numpy(), 
            F.exp(A).detach().numpy(), atol=1e-5, rtol=1e-5)

    torch.exp(TA).sum().backward()
    F.exp(A).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_relu(shape, device):
    _A = np.random.randn(*shape).astype(np.float32) + 5.
    A = thanos.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    np.testing.assert_allclose(
            torch.relu(TA).detach().numpy(), 
            F.relu(A).detach().numpy(), atol=1e-5, rtol=1e-5)

    torch.relu(TA).sum().backward()
    F.relu(A).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_sqrt(shape, device):
    _A = np.random.randn(*shape).astype(np.float32) + 5.
    A = thanos.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    np.testing.assert_allclose(
            torch.sqrt(TA).detach().numpy(), 
            F.sqrt(A).detach().numpy(), atol=1e-5, rtol=1e-5)

    torch.sqrt(TA).sum().backward()
    F.sqrt(A).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)

STACK_PARAMETERS = [
    ((5, 5), 0, 1),
    ((5, 5), 0, 2),
    ((1,5,7), 2, 5)]
@pytest.mark.parametrize("shape, axis, l", STACK_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_stack(shape, axis, l, device):
    _A = [np.random.randn(*shape).astype(np.float32) for i in range(l)]
    A = [thanos.Tensor(_A[i], device=device) for i in range(l)]
    TA = [torch.Tensor(_A[i]) for i in range(l)]
    for torch_a in TA:
        torch_a.requires_grad = True
    np.testing.assert_allclose(
            torch.stack(TA, dim=axis).detach().numpy(), 
            F.stack(A, axis=axis).detach().numpy(), atol=1e-5, rtol=1e-5)

    torch.stack(TA, dim=axis).sum().backward()
    F.stack(A, axis=axis).sum().backward()
    for i in range(l):
        np.testing.assert_allclose(TA[i].grad.numpy(), A[i].grad.numpy(), atol=1e-5, rtol=1e-5)


BROADCAST_SHAPES = [
    ((1, 1, 1), (3, 3, 3)),
    ((4, 1, 6), (4, 3, 6))]
@pytest.mark.parametrize("shape,shape_to", BROADCAST_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_broadcast_to(shape, shape_to, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = thanos.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    np.testing.assert_allclose(
            torch.broadcast_to(TA, shape_to).detach().numpy(), 
            F.broadcast_to(A, shape_to).detach().numpy(), atol=1e-5, rtol=1e-5)

    torch.broadcast_to(TA, shape_to).sum().backward()
    F.broadcast_to(A, shape_to).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)

RESHAPE_SHAPES = [
    ((1, 1, 1), (1,)),
    ((4, 1, 6), (6, 4, 1))]
@pytest.mark.parametrize("shape,shape_to", RESHAPE_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_reshape(shape, shape_to, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = thanos.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    np.testing.assert_allclose(
            torch.reshape(TA, shape_to).detach().numpy(), 
            F.reshape(A, shape_to).detach().numpy(), atol=1e-5, rtol=1e-5)

    torch.reshape(TA, shape_to).sum().backward()
    F.reshape(A, shape_to).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)

TRANSPOSE_SHAPES = [(1, 1, 1), (4, 5, 6)]
TRANSPOSE_AXES = [(0, 1), (0, 2), None]
@pytest.mark.parametrize("shape", TRANSPOSE_SHAPES)
@pytest.mark.parametrize("axes", TRANSPOSE_AXES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_transpose(shape, axes, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = thanos.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    if axes is None:
        axes = (-1, -2)
    np.testing.assert_allclose(
            torch.transpose(TA, axes[0], axes[1]).detach().numpy(), 
            F.transpose(A, axes).detach().numpy(), atol=1e-5, rtol=1e-5)

    torch.transpose(TA, axes[0], axes[1]).sum().backward()
    F.transpose(A, axes).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("shape, axes", SUMMATION_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_logsumexp(shape, axes, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = thanos.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    if axes is None:
        t_axes = tuple(list(range(len(shape))))
    else:
        t_axes = axes
    np.testing.assert_allclose(
            torch.logsumexp(TA, dim=t_axes).detach().numpy(), 
            F.logsumexp(A, axes).detach().numpy(), atol=1e-5, rtol=1e-5)

    torch.logsumexp(TA, dim=t_axes).sum().backward()
    F.logsumexp(A, axes).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)

# TODO need to flatten the syntax between PyTorch and my code.
@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_equal(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    _B = np.random.randn(*shape).astype(np.float32)
    A = thanos.Tensor(_A, device=device)
    B = thanos.Tensor(_B, device=device)
    C = thanos.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    TB = torch.Tensor(_B)
    TB.requires_grad = True
    TC = torch.Tensor(_A)
    TC.requires_grad = True
    np.testing.assert_allclose((TA == TB).detach().numpy(), F.equal(A, B).detach().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose((TA == TC).detach().numpy(), F.equal(A, C).detach().numpy(), atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("shape, axes", SUMMATION_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_max(shape, axes, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = thanos.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    t_axes = axes
    if axes is None:
        t_axes = -1
    np.testing.assert_allclose(
            torch.max(TA, dim=t_axes)[0].detach().numpy(), 
            F.max(A, axis=axes).detach().numpy(), atol=1e-5, rtol=1e-5)

    torch.max(TA, dim=t_axes)[0].sum().backward()
    F.max(A, axis=axes).sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)

if __name__ == "__main__":
    pytest.main()
