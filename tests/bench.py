import sys
sys.path.append('./python')
import itertools
import numpy as np
import pytest
import torch

import thanos
import thanos.nn.functional as F
#from thanos import backend_ndarray as nd
import time

def test_reduce_sum():
    def sum_test(A, axis, need_print=False):
        start_time = time.time()
        ans = F.summation(A, axis=axis)
        if need_print:
            print(ans.shape)
        return time.time() - start_time

    shapes = [(8, 4, 5), (512, 512), (512 * 64, 8), (8, 512 * 64), (8, 512 * 64)]
    #shapes = [(8, 3, 2)]
    axis = [1, 1, 1, 0, 0]
    #shapes = [(3, 4)]
    for idx in range(len(shapes)):
        shape = shapes[idx]
        print(shape)
        all_time = 0
        for i in range(100):
            _A = np.random.randn(*shape).astype(np.float32)
            A = thanos.Tensor(_A, device=thanos.cuda())
            #TA = torch.tensor(_A)
            #print(A)
            #print(torch.sum(TA, dim=axis[idx]))
            all_time += sum_test(A, axis[idx])
        sum_test(A, axis[idx], need_print=True)
        print(all_time)
        print("==================")

def test_matmul():
    def matmul_test(A, B):
        start_time = time.time()
        ans = A @ B
        return time.time() - start_time

    shapes_a = [(4096, 1024), (512, 512), (512 * 64, 8), (8, 512 * 64), (8, 512 * 64)]
    shapes_b = [(1024, 8192), (512, 512), (8, 512 * 64), (512 * 64, 8), (512 * 64, 8)]
    #shapes = [(8, 3, 2)]
    axis = [1, 1, 1, 0, 0]
    #shapes = [(3, 4)]
    for idx in range(len(shapes_a)):
        shape_a = shapes_a[idx]
        shape_b = shapes_b[idx]
        print(shape_a, shape_b)
        all_time = 0
        for i in range(10):
            _A = np.random.randn(*shape_a).astype(np.float32)
            A = thanos.Tensor(_A, device=thanos.cuda())
            _B = np.random.randn(*shape_b).astype(np.float32)
            B = thanos.Tensor(_B, device=thanos.cuda())
            #TA = torch.tensor(_A)
            #print(A)
            #print(torch.sum(TA, dim=axis[idx]))
            all_time += matmul_test(A, B)
        print(all_time)
        print("==================")

def test_matmul_torch():
    def matmul_test(A, B):
        start_time = time.time()
        ans = A @ B
        return time.time() - start_time

    shapes_a = [(4096, 1024), (512, 512), (512 * 64, 8), (8, 512 * 64), (8, 512 * 64)]
    shapes_b = [(1024, 8192), (512, 512), (8, 512 * 64), (512 * 64, 8), (512 * 64, 8)]
    #shapes = [(8, 3, 2)]
    axis = [1, 1, 1, 0, 0]
    #shapes = [(3, 4)]
    for idx in range(len(shapes_a)):
        shape_a = shapes_a[idx]
        shape_b = shapes_b[idx]
        print(shape_a, shape_b)
        all_time = 0
        for i in range(10):
            _A = np.random.randn(*shape_a).astype(np.float32)
            A = torch.tensor(_A).to(torch.device("cuda"))
            _B = np.random.randn(*shape_b).astype(np.float32)
            B = torch.tensor(_B).to(torch.device("cuda"))
            #TA = torch.tensor(_A)
            #print(A)
            #print(torch.sum(TA, dim=axis[idx]))
            all_time += matmul_test(A, B)
        print(all_time)
        print("==================")
test_matmul()
test_matmul_torch()
