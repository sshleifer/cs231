import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


def train_torchnet(X, y, batch_size=100, n_batches=1000,weight_decay=1e-3,
                   **model_kwargs):
    # Xbatch = np.array_split(X, int(X/batch_size))
    # ybatch = np.array_split(y, int(X/batch_size))
    model = ThreeLayerTorchNet(**model_kwargs)
    train_scores = []
    val_scores = []
    optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
    for step in tqdm(range(n_batches)):
        offset = (step * batch_size) % (y.shape[0] - batch_size)
        xbatch = X[offset:(offset + batch_size)]
        ybatch = y[offset:(offset + batch_size)]
        xt = Variable(torch.FloatTensor(xbatch))
        yt = Variable(torch.LongTensor(ybatch))
        optimizer.zero_grad()
        # print xt.data.shape
        output = model(xt)
        loss = F.nll_loss(F.log_softmax(output), yt)
        loss.backward()
        optimizer.step()
        #     if step % 50 == 0:
        #         train_scores.append(model.funcy_scorer(X,y))
        #         val_scores.append(model.funcy_scorer(data['X_val'], data['y_val']))
    return model


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

    def funcy_scorer(self, X, y):
        probas = self.forward(Variable(torch.FloatTensor(X)))
        y = Variable(torch.LongTensor(y))
        probas, indices = torch.max(probas.data, 1)
        return y.data.eq(indices).sum() / float(y.size(0))


def train(epoch, X, y, batch_size=64):
    raise NotImplementedError
