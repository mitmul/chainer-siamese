#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('Agg')
import argparse
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from chainer import cuda, optimizers
from SiameseNetwork import SiameseNetwork


def get_data():
    # data preparation
    mnist = fetch_mldata('MNIST original')
    data = mnist['data'].astype(np.float32)
    label = mnist['target'].astype(np.int32)
    N = 60000  # of training data
    N_test = 10000  # of test data
    train_data = data[:N].reshape((N, 1, 28, 28)) / 255.0
    test_data = data[N:].reshape((N_test, 1, 28, 28)) / 255.0
    train_label = label[:N]
    test_label = label[N:]

    return train_data, train_label, test_data, test_label


def get_model_optimizer(args):
    # model preparation
    model = SiameseNetwork()
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # optimizer
    if args.optimizer == 'SGD':
        optimizer = optimizers.MomentumSGD(lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'Adam':
        optimizer = optimizers.Adam()
    optimizer.setup(model)

    return model, optimizer


def train(args, train_data, train_label, model, optimizer):
    xp = cuda.cupy if args.gpu >= 0 else np
    N = train_data.shape[0]
    losses = []
    for epoch in range(1, args.epoch + 1):
        mean_loss = 0
        perm = np.random.permutation(N)
        for i in range(0, N, args.batchsize):
            x0_batch = train_data[perm[i:i + args.batchsize]]
            x1_batch = train_data[perm[i:i + args.batchsize]][::-1]
            y0_batch = train_label[perm[i:i + args.batchsize]]
            y1_batch = train_label[perm[i:i + args.batchsize]][::-1]
            label = xp.array(y0_batch == y1_batch, dtype=np.int32)

            x0_batch = xp.asarray(x0_batch, dtype=np.float32)
            x1_batch = xp.asarray(x1_batch, dtype=np.float32)

            optimizer.zero_grads()
            loss = model.forward(x0_batch, x1_batch, label)
            loss.backward()
            optimizer.update()

            mean_loss += float(loss.data) * args.batchsize

        if args.optimizer == 'SGD':
            optimizer.lr = args.lr * (1 + args.gamma * epoch) ** -args.power
        losses.append(mean_loss / N)

        plt.clf()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(losses)
        plt.savefig('loss.png')

        print('epoch:{}/{}\tlr:{}\tloss:{}'.format(
            epoch, args.epoch, optimizer.lr, mean_loss / N))

        if epoch % 10 == 0:
            pickle.dump(model, open('model_{}.pkl'.format(epoch), 'wb'), -1)

    return model


def test(args, model, test_data, test_label):
    xp = cuda.cupy if args.gpu >= 0 else np
    N = test_data.shape[0]
    results = xp.empty((test_data.shape[0], 2))
    for i in range(0, N, args.batchsize):
        x_batch = test_data[i:i + args.batchsize]
        x_batch = xp.asarray(x_batch, dtype=xp.float32)
        y = model.forward_once(x_batch, train=False)
        results[i:i + args.batchsize] = y.data

    if args.gpu >= 0:
        results = xp.asnumpy(results)

    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    for i in range(10):
        feat = results[np.where(test_label == i)]
        plt.plot(feat[:, 0], feat[:, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    plt.savefig('result_{}.png'.format(args.epoch))

if __name__ == '__main__':
    # prep for args
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batchsize', default=128, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--gamma', default=0.001, type=float)
    parser.add_argument('--power', default=0.75, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--optimizer', default='SGD', type=str,
                        choices=['SGD', 'Adam'])
    args = parser.parse_args()

    if args.train == 1:
        train_data, train_label, test_data, test_label = get_data()
        model, optimizer = get_model_optimizer(args)
        model = train(args, train_data, train_label, model, optimizer)
        pickle.dump(model, open('model_{}.pkl'.format(args.epoch), 'wb'), -1)
    else:
        train_data, train_label, test_data, test_label = get_data()
        model = pickle.load(open('model_{}.pkl'.format(args.epoch)))
        test(args, model, test_data, test_label)
