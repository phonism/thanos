import sys
sys.path.append('./python')
import itertools
import numpy as np
import pytest
import torch
import thanos.nn.functional as F

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

@pytest.mark.parametrize("shape", SOFTMAX_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_layernorm(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = thanos.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    norm = thanos.nn.LayerNorm(shape[-1])
    if device == thanos.cuda():
        norm.cuda()
    C = norm(A)
    TC = torch.nn.LayerNorm(shape[-1])(TA)
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
    (8, 100, 32),
]
@pytest.mark.parametrize("shape", ATTENTION_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_onehead_attention(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = thanos.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True

    attn = thanos.nn.MultiheadAttention(shape[2], 1)

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
    (8, 100, 32),
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

@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_embedding(device):
    num_embeddings = 1000
    embedding_dim = 32

    _A = np.array([[5, 6], [3, 4]]).astype(np.int64)
    A = thanos.Tensor(_A, device=device)
    TA = torch.LongTensor(_A)

    embed = thanos.nn.Embedding(num_embeddings, embedding_dim)
    torch_embed = torch.nn.Embedding(num_embeddings, embedding_dim)
    embed.weight = thanos.nn.Parameter(torch_embed.weight.detach().numpy())

    if device == thanos.cuda():
        embed.cuda()

    thanos_out = embed(A)
    torch_out = torch_embed(TA)

    np.testing.assert_allclose(
            thanos_out.detach().numpy(), 
            torch_out.detach().numpy(), 
            atol=1e-5, rtol=1e-5)

    thanos_out.sum().backward()
    torch_out.sum().backward()
    np.testing.assert_allclose(embed.weight.detach().numpy(), torch_embed.weight.detach().numpy(), atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("shape", SOFTMAX_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_silu(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = thanos.Tensor(_A, device=device)
    TA = torch.Tensor(_A)
    TA.requires_grad = True
    C = thanos.nn.SiLU()(A)
    TC = torch.nn.SiLU()(TA)
    np.testing.assert_allclose(TC.detach().numpy(), C.detach().numpy(), atol=1e-5, rtol=1e-5)

    C.sum().backward()
    TC.sum().backward()
    np.testing.assert_allclose(TA.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main()
