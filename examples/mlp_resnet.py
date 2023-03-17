import sys
sys.path.append('../python')
import thanos.data as data
import thanos.optim as optim
import thanos.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    module = nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim)
            )
        ),
        nn.ReLU()
    )
    return module


def MLPResNet(dim, hidden_dim=128, num_blocks=10, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    modules = [
        nn.Linear(dim, hidden_dim),
        nn.ReLU()
    ]
    for _ in range(num_blocks):
        modules.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm=norm, drop_prob=drop_prob))
    modules.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*modules)

def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    hit, total = 0, 0
    loss_func = nn.SoftmaxLoss()
    loss_all = 0
    if opt is not None:
        model.train()
        for idx, data in enumerate(dataloader):
            x, y = data
            output = model(x)
            opt.reset_grad()
            loss = loss_func(output, y)
            loss_all += loss.numpy()
            loss.backward()
            opt.step()
            hit += (y.numpy() == output.numpy().argmax(1)).sum()
            total += y.shape[0]
    else:
        model.eval()
        for idx, data in enumerate(dataloader):
            x, y = data
            output = model(x)
            loss = loss_func(output, y)
            loss_all += loss.numpy()
            hit += (y.numpy() == output.numpy().argmax(1)).sum()
            total += y.shape[0]
    acc = hit / total
    return acc, loss_all / (idx + 1)

def train_mnist(
        batch_size=128, epochs=10, optimizer=optim.Adam,
        lr=0.001, weight_decay=0.001, hidden_dim=128, data_dir="data"):
    np.random.seed(4)
    train_img_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_label_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    test_img_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_label_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
    train_dataset = data.MNISTDataset(train_img_path, train_label_path)
    test_dataset = data.MNISTDataset(test_img_path, test_label_path)

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = data.DataLoader(test_dataset)
    model = MLPResNet(784, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    start_time = time.time()
    for idx in range(epochs):
        train_acc, train_loss = epoch(train_dataloader, model, opt)
        #print(idx, " DONE: train_acc:", train_acc, " train_loss:", train_loss, " tensor_counter:", thanos.autograd.TENSOR_COUNTER, " duration:", time.time() - start_time)
        print(idx, " DONE: train_acc:", train_acc, " train_loss:", train_loss, " duration:", time.time() - start_time)
        start_time = time.time()
    test_acc, test_loss = epoch(test_dataloader, model)
    return (train_acc, train_loss, test_acc, test_loss)

if __name__ == "__main__":
    train_acc, train_loss, test_acc, test_loss = train_mnist(data_dir="./data")
    print(train_acc, train_loss, test_acc, test_loss)
