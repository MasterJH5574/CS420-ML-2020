import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_data2(path):
    # load all MNIST data
    fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_X = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)
    fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_Y = loaded[8:].reshape(60000).astype(np.float)
    fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_X = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)
    fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_Y = loaded[8:].reshape(10000).astype(np.float)

    #visualiza data
    sample_num = 8
    num_classes = 10
    for y in range(num_classes):
        idxs = np.flatnonzero(train_Y == y)
        idxs = np.random.choice(idxs, sample_num, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(sample_num, num_classes, plt_idx)
            plt.imshow(train_X[idx, :, :, :].reshape((28,28)),cmap=plt.cm.gray)
            plt.axis('off')
            if i == 0:
                plt.title(y)
    plt.show()

    # reshaple into rows and normaliza
    train_X = train_X.reshape((train_X.shape[0], -1))
    test_X = test_X.reshape((test_X.shape[0], -1))
    mean_image = np.mean(train_X, axis=0)
    train_X = train_X - mean_image
    test_X = test_X - mean_image

    # add a bias columu into X
    train_X = np.hstack([train_X, np.ones((train_X.shape[0], 1))])
    test_X = np.hstack([test_X, np.ones((test_X.shape[0], 1))])
    train_Y = train_Y.astype(np.int32)
    test_Y = test_Y.astype(np.int32)
    return train_X, train_Y, test_X, test_Y


X_train, Y_train, X_test, Y_test = load_data2('/home/kesci/input/MNIST_dataset4284')