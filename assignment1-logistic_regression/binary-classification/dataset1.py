import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_data1(path):
    # load all MNIST data
    fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_X_all = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)
    fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_Y_all = loaded[8:].reshape(60000).astype(np.float)
    fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_X_all = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)
    fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_Y_all = loaded[8:].reshape(10000).astype(np.float)

    # subsample data
    train_idxs_df = pd.read_csv(os.path.join(path, 'train_indices.csv'))
    test_idxs_df = pd.read_csv(os.path.join(path, 'test_indices.csv'))
    pos_train_indices = train_idxs_df['pos_train_indices'].tolist()
    neg_train_indices = train_idxs_df['neg_train_indices'].tolist()
    pos_test_indices = test_idxs_df['pos_test_indices'].tolist()
    neg_test_indices = test_idxs_df['neg_test_indices'].tolist()
    train_Y_all[pos_train_indices] = 1
    train_Y_all[neg_train_indices] = 0
    test_Y_all[pos_test_indices] = 1
    test_Y_all[neg_test_indices] = 0
    train_indices = np.append(pos_train_indices, neg_train_indices)
    test_indices = np.append(pos_test_indices, neg_test_indices)
    train_X = train_X_all[train_indices]
    train_Y = train_Y_all[train_indices]
    test_X = test_X_all[test_indices]
    test_Y = test_Y_all[test_indices]

    # visualiza data
    sample_num = 8
    pos_sample_indices = np.random.choice(pos_train_indices, sample_num, replace=False)
    neg_sample_indices = np.random.choice(neg_train_indices, sample_num, replace=False)
    for i, idx in enumerate(pos_sample_indices):
        plt_idx = i + 1
        plt.subplot(2, sample_num, plt_idx)
        plt.imshow(train_X_all[idx, :, :, :].reshape((28, 28)), cmap=plt.cm.gray)
        plt.axis('off')
        if i == 0:
            plt.title('Positive')

    for i, idx in enumerate(neg_sample_indices):
        plt_idx = sample_num + i + 1
        plt.subplot(2, sample_num, plt_idx)
        plt.imshow(train_X_all[idx, :, :, :].reshape((28, 28)), cmap=plt.cm.gray)
        plt.axis('off')
        if i == 0:
            plt.title('Negative')

    # reshaple into rows and normaliza
    train_X = train_X.reshape((train_X.shape[0], -1))
    test_X = test_X.reshape((test_X.shape[0], -1))
    mean_image = np.mean(train_X, axis=0)
    train_X = train_X - mean_image
    test_X = test_X - mean_image

    # add a bias columu into X
    train_X = np.hstack([train_X, np.ones((train_X.shape[0], 1))])
    test_X = np.hstack([test_X, np.ones((test_X.shape[0], 1))])
    return train_X, train_Y, test_X, test_Y


X_train, Y_train, X_test, Y_test = load_data1('/home/kesci/input/MNIST_dataset4284')