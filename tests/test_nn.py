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
    norm = thanos.nn.BatchNorm1d(shape[1])
    if device == thanos.cuda():
        norm.cuda()
    C = norm(A)
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

ATTENTION_SHAPES = [
    (8, 32, 64),
]
@pytest.mark.parametrize("shape", ATTENTION_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_attention(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = thanos.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True

    attn = thanos.nn.Attention()
    #attn(A)

    torch_attn = torch.nn.MultiheadAttention(shape[2], 1, bias=False, batch_first=True)
    attn.w_kqv = thanos.nn.Parameter(torch_attn.in_proj_weight.detach().numpy().T)
    attn.w_out = thanos.nn.Parameter(torch_attn.out_proj.weight.detach().numpy().T)
    M = torch.triu(-float("inf") * torch.ones(shape[1], shape[1]), 1)

    if device == thanos.cuda():
        attn.cuda()
    thanos_out = attn(A)
    torch_out = torch_attn(TA, TA, TA, attn_mask=M)

    np.testing.assert_allclose(
            thanos_out[0].detach().numpy(), 
            torch_out[0].detach().numpy(), 
            atol=1e-5, rtol=1e-5)

    thanos_out[0].sum().backward()
    torch_out[0].sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)


ATTENTION_SHAPES = [
    (8, 32, 64),
]
@pytest.mark.parametrize("shape", ATTENTION_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_multihead_attention(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = thanos.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True

    attn = thanos.nn.MultiheadAttention(dim=shape[2], heads=4)

    torch_attn = torch.nn.MultiheadAttention(shape[2], 4, bias=False, batch_first=True)
    attn.w_kqv = thanos.nn.Parameter(torch_attn.in_proj_weight.detach().numpy().T)
    attn.w_out = thanos.nn.Parameter(torch_attn.out_proj.weight.detach().numpy().T)
    M = torch.triu(-float("inf") * torch.ones(shape[1], shape[1]), 1)

    if device == thanos.cuda():
        attn.cuda()
    thanos_out = attn(A)
    torch_out = torch_attn(TA, TA, TA, attn_mask=M)

    np.testing.assert_allclose(
            thanos_out[0].detach().numpy(), 
            torch_out[0].detach().numpy(), 
            atol=1e-5, rtol=1e-5)

    thanos_out[0].sum().backward()
    torch_out[0].sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main()
