import os
import time
import numpy as np
import torch
import torch as thanos
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import sys


sys.path.append("../python")
import thanos.data as data

np.random.seed(0)
use_cuda = thanos.cuda.is_available()

import triton
import triton.language as tl

@triton.jit
def add_kernel(X, Y, Z, N):
    pid = tl.program_id(0)
    offsets = pid * 1024 + tl.arange(0, 1024)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    z = x + y
    tl.store(Z + offsets, z, mask=mask)

def triton_add(x, y):
    assert x.is_contiguous() and y.is_contiguous(), "Tensors must be contiguous"
    assert x.shape == y.shape, "Shapes of the tensors must match"
    
    z = torch.empty_like(x)
    N = x.numel()
    
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, z, N, BLOCK_SIZE=1024)
    
    return z

original_add = torch.Tensor.__add__

def custom_add(self, other):
    if isinstance(other, torch.Tensor):
        print("YOU")
        return triton_add(self, other)
    else:
        print("FUCK")
        return original_add(self, other)

# 替换 `__add__` 方法
torch.Tensor.__add__ = custom_add
torch.add = custom_add

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    class Block(nn.Module):
        def __init__(self):
            super(Block, self).__init__()
            self.fc1 = nn.Linear(dim, hidden_dim)
            self.norm1 = norm(hidden_dim)
            self.drop = nn.Dropout(p=drop_prob)
            self.fc2 = nn.Linear(hidden_dim, dim)
            self.norm2 = norm(dim)
            self.relu = nn.ReLU()

        def forward(self, x):
            identity = x
            out = self.fc1(x)
            out = self.norm1(out)
            out = self.relu(out)
            out = self.drop(out)
            out = self.fc2(out)
            out = self.norm2(out)
            out += identity
            out = self.relu(out)
            return out

    return Block()

def MLPResNet(input_dim, hidden_dim=128, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(num_blocks):
        layers.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm=norm, drop_prob=drop_prob))
    layers.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*layers)

def epoch(dataloader, model, opt=None):
    thanos.manual_seed(4)
    hit, total = 0, 0
    loss_func = nn.CrossEntropyLoss()
    total_loss = 0

    for x, y in dataloader:
        if use_cuda:
            x, y = x.cuda(), y.cuda()

        x = x.view(x.size(0), -1)
        output = model(x)
        loss = loss_func(output, y)
        total_loss += loss.item()

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        hit += (y == output.argmax(1)).sum().item()
        total += y.size(0)

    accuracy = hit / total
    average_loss = total_loss / len(dataloader)
    return accuracy, average_loss

def train_mnist(batch_size=128, epochs=1, optimizer=optim.AdamW, lr=0.001, weight_decay=0.001, hidden_dim=128, data_dir="data"):
    thanos.manual_seed(4)

    train_dataset = data.MNISTDataset(os.path.join(data_dir, "train-images-idx3-ubyte.gz"), 
                                      os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    test_dataset = data.MNISTDataset(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"), 
                                     os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset)

    model = MLPResNet(784, hidden_dim=hidden_dim)
    if use_cuda:
        model.cuda()

    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    start_time = time.time()
    for idx in range(epochs):
        train_acc, train_loss = epoch(train_loader, model, opt)
        print(idx, " DONE: train_acc:", train_acc, " train_loss:", train_loss, time.time() - start_time)
        start_time = time.time()
    #test_acc, test_loss = epoch(test_dataloader, model)
    test_acc = 0.0
    test_loss = 0.
    return (train_acc, train_loss, test_acc, test_loss)

if __name__ == "__main__":
    train_acc, train_loss, test_acc, test_loss = train_mnist(data_dir="./data")
    print(train_acc, train_loss, test_acc, test_loss)
