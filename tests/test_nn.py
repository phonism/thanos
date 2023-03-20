import sys
sys.path.append('./python')
import itertools
import numpy as np
import pytest
import torch

import thanos
#from thanos import backend_ndarray as nd

_DEVICES = [
    thanos.cpu(), 
    pytest.param(
            thanos.cuda(), 
            marks=pytest.mark.skipif(not thanos.cuda().enabled(), reason="No GPU"))]

ATTENTION_SHAPES = [
    (8, 32, 64),
    (8, 32, 64),
]
@pytest.mark.parametrize("shape", ATTENTION_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_attention(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = thanos.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True

    atten = thanos.nn.Attention()
    atten(A)

SOFTMAX_SHAPES = [
        (8, 64),
        (8, 32, 64),
]
@pytest.mark.parametrize("shape", SOFTMAX_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_softmax(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = thanos.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    C = thanos.nn.Softmax(dim=-1)(A)
    TC = torch.nn.Softmax(dim=-1)(TA)
    np.testing.assert_allclose(TC.detach().numpy(), C.detach().numpy(), atol=1e-5, rtol=1e-5)

    C.sum().backward()
    TC.sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)


NN_BASE_SHAPES = [
        (8, 64),
        (32, 32),
]
@pytest.mark.parametrize("shape", NN_BASE_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_batchnorm1d(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = thanos.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    C = thanos.nn.BatchNorm1d(shape[1])(A)
    TC = torch.nn.BatchNorm1d(shape[1])(TA)
    np.testing.assert_allclose(TC.detach().numpy(), C.detach().numpy(), atol=1e-5, rtol=1e-5)

    C.sum().backward()
    TC.sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("shape", NN_BASE_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_relu(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = thanos.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    C = thanos.nn.ReLU()(A)
    TC = torch.nn.ReLU()(TA)
    np.testing.assert_allclose(TC.detach().numpy(), C.detach().numpy(), atol=1e-5, rtol=1e-5)

    C.sum().backward()
    TC.sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main()
