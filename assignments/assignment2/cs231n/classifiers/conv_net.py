from __future__ import print_function, division

import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

USE_CUDA = torch.cuda.is_available()

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def batch_xy(X, y, n=100):
    return zip(batch(X, n=n), batch(y, n=n))


def train_torchnet(X, y, batch_size=100, n_batches=np.inf, weight_decay=1e-3,
                   print_every=0, X_val=None, y_val=None, num_epochs=1,
                   use_cuda=USE_CUDA, model=None, **model_kwargs):
    # Xbatch = np.array_split(X, int(X/batch_size))
    # ybatch = np.array_split(y, int(X/batch_size))
    if model is None:
        model = ThreeLayerTorchNet(**model_kwargs)
        if use_cuda:
            model = model.cuda()
            model.cuda_flag = True

    optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
    #for step in tqdm(range(n_batches)):
    step = model.max_step
    for epoch in tqdm(range(num_epochs)):
        # bprint(epoch, step)
        for xbatch, ybatch in batch_xy(X, y, batch_size):
            step += 1
            if step >= n_batches:
                print('enough batches, returning')
                return model
            xt = Variable(torch.FloatTensor(xbatch))
            yt = Variable(torch.LongTensor(ybatch))
            if use_cuda:
                xt = xt.cuda()
                yt = yt.cuda()
            optimizer.zero_grad()
            # print xt.data.shape
            output = model(xt)
            loss = F.nll_loss(F.log_softmax(output), yt)
            loss.backward()
            optimizer.step()
            if print_every and step % print_every == 0:
                score = model.funcy_scorer(X_val, y_val)
                print(score)
                model.val_scores[step] = score
                model.max_step = step
                model.train_loss[step] = loss.cpu().data.numpy()[0]
                if use_cuda:
                    loss = loss.cuda()
    return model

def make_conv_block(num_filters, filter_size=5,
                    C=3, padding=1, pool_height=2):
    layers = []

    conv = nn.Conv2d(C, num_filters, filter_size,
                     stride=1, padding=padding)
    layers = [conv,
              nn.BatchNorm2d(num_filters),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(pool_height)]
    return nn.Sequential(*layers)



class ThreeLayerTorchNet(nn.Module):
    cuda_flag = False
    batch_size = 100
    max_step = 0



    def __init__(self, num_filters=32, filter_size=7, padding=3,
                 pool_height=2,
                 input_dim=(3, 32, 32), hidden_dim=100, num_classes=10):
        super(ThreeLayerTorchNet, self).__init__()
        self.val_scores = {}
        self.train_loss = {}
        # input_dim  = (49000, 3, 32, 32)
        self.pool_height = pool_height
        C, H, W = input_dim
        self.conv1 = make_conv_block(num_filters, filter_size, padding=padding,
                                     pool_height=pool_height)
        Hout = (input_dim[1] - filter_size + 2 * padding) + 1

        self.output_dim = int((num_filters * Hout / pool_height * Hout / pool_height))
        #self.batch_norm1 = torch.nn.BatchNorm2d(num_filters)
        #self.conv2 = nn.Conv2D

        print(Hout, self.output_dim)
        self.affine1 = nn.Linear(self.output_dim, hidden_dim)
        self.batch_norm_fc = torch.nn.BatchNorm1d(hidden_dim)
        self.affine2 = nn.Linear(hidden_dim, num_classes)


    def forward(self, x):
        x = self.conv1(x)

        x = x.view(-1, self.output_dim)
        x = F.relu(self.batch_norm_fc(self.affine1(x)))
        x = F.dropout(x, training=self.training)
        x = self.affine2(x)
        return x

    def funcy_scorer(self, X, y):
        correct = 0
        total = 0
        self.training = False
        for xbatch, ybatch in batch_xy(X, y, self.batch_size):

            xt = Variable(torch.FloatTensor(xbatch))
            yt = Variable(torch.LongTensor(ybatch))
            if self.cuda_flag:
                xt = xt.cuda()
                yt = yt.cuda()
            probas = self.forward(xt)
            probas, indices = torch.max(probas.data, 1)
            correct += yt.data.eq(indices).sum()
            total += float(yt.size(0))
        self.training = True
        return correct/total

    def custom_scorer(self, data):
        return {
            # 'train': self.funcy_scorer(data['X_train'], data['y_train']),
            'val': self.funcy_scorer(data['X_val'], data['y_val'])
        }


def train(epoch, X, y, batch_size=64):
    raise NotImplementedError


'''
modelavg = train_torchnet(data['X_train'], data['y_train'],
                             num_epochs=5,
                             # n_batches=10,
                             weight_decay=.001,
                             print_every=500,
                             X_val=data['X_val'], y_val=data['y_val'],
                             filter_size=5,
                             num_filters=64,
                             use_cuda=True)
'''