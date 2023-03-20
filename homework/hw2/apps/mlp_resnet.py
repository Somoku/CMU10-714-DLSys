import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(nn.Residual(
        nn.Sequential(nn.Linear(dim, hidden_dim), norm(hidden_dim),
                      nn.ReLU(), nn.Dropout(drop_prob),
                      nn.Linear(hidden_dim, dim), norm(dim))), nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(),
                         *[ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim//2, 
                                         norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)],
                         nn.Linear(hidden_dim, num_classes))
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_sum = []
    err_rate_sum = []
    softmaxloss = nn.SoftmaxLoss()
    sample_num = len(dataloader.dataset)
    if opt:
        model.train()
        for X, y in dataloader:
            opt.reset_grad()
            h = model(X)
            err_rate_sum.append(np.sum(h.numpy().argmax(axis=1) != y.numpy()))
            loss = softmaxloss(h, y)
            loss_sum.append(loss.numpy())
            loss.backward()
            opt.step()
    else:
        model.eval()
        for X, y in dataloader:
            h = model(X)
            loss = softmaxloss(h, y)
            err_rate_sum.append(np.sum(h.numpy().argmax(axis=1) != y.numpy()))
            loss_sum.append(loss.numpy())
    return np.sum(err_rate_sum) / sample_num, np.average(loss_sum)
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_set = ndl.data.MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz",
                                      f"{data_dir}/train-labels-idx1-ubyte.gz")
    test_set = ndl.data.MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz",
                                      f"{data_dir}/t10k-labels-idx1-ubyte.gz")
    train_loader = ndl.data.DataLoader(train_set, batch_size, shuffle=True)
    test_loader = ndl.data.DataLoader(test_set, batch_size)
    model = MLPResNet(28 * 28, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        train_err, train_loss = epoch(train_loader, model, opt)
    test_err, test_loss = epoch(test_loader, model)
    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
