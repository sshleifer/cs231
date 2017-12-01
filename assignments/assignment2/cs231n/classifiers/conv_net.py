import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class ThreeLayerTorchNet(nn.Module):
    def __init__(self, num_filters=32, filter_size=7,
                 input_dim=(3, 32, 32), hidden_dim=100, num_classes=32):
        super(ThreeLayerTorchNet, self).__init__()
        # input_dim  = (49000, 3, 32, 32)
        pool_height = 2
        C, H, W = input_dim
        self.conv1 = nn.Conv2d(C, num_filters, filter_size,
                               stride=1, padding=3)
        # self.max_pool= nn.MaxPool2d(pool_height)
        output_dim = (32 * 32 / pool_height * 32 / pool_height) / 4.
        self.affine1 = nn.Linear(output_dim, hidden_dim)
        self.affine2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.affine1(x)
        x = nn.ReLU(x)
        x = self.affine2(x)
        return x

def train(epoch, X, y, batch_size=64):
    Xbatch = np.array_split(X, int(X/batch_size))
    ybatch = np.array_split(y, int(X/batch_size))

    for batch_idx, (xbatch, target) in enumerate(zip(Xbatch, ybatch)):
        x, y = Variable(xbatch), Variable(ybatch)


import torch.optim as optim
import torch.nn.functional as F


class ThreeLayerTorchNet(nn.Module):
    def __init__(self, num_filters=32, filter_size=7,
                 input_dim=(3, 32, 32), hidden_dim=100, num_classes=10):
        super(ThreeLayerTorchNet, self).__init__()
        # input_dim  = (49000, 3, 32, 32)
        pool_height = 2
        C, H, W = input_dim
        self.conv1 = nn.Conv2d(C, num_filters, filter_size, stride=1, padding=3)
        # self.max_pool= nn.MaxPool2d(pool_height)
        self.output_dim = (32 * 32 / pool_height * 32 / pool_height)
        self.affine1 = nn.Linear(self.output_dim, hidden_dim)
        self.affine2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.output_dim)
        x = F.relu(self.affine1(x))
        x = self.affine2(x)
        return x

    def funcy_scorer(seprilf, X, y):
        probas = self.forward(Variable(torch.FloatTensor(X)))
        y = Variable(torch.LongTensor(y))
        probas, indices = torch.max(probas.data, 1)
        return y.data.eq(indices).sum() / float(y.size(0))


def train(epoch, X, y, batch_size=64):
    raise NotImplementedError
