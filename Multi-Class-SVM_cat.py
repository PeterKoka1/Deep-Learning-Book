"""
Computing L2 loss from first iteration of multi-class SVM
using a linear classifier.

Weights were all set to 0.1.
The train set was 5 images: a cat, person, fire hydrant, book, and computer.
"""

import numpy as np
import os
from scipy.misc import imread
import scipy

train_dir = "C:/Users/PeterKokalov/lpthw/StanfordCNN/Lecture2/KNN/images/train"

images = []

def compress(path):
    N_N = scipy.misc.imread(path)
    N_1 = np.ravel(N_N)
    X_i = np.reshape(N_1, (len(N_1), 1))

    return X_i


def create_X_y():
    Xtr = np.array([])
    ytr = np.array([])

    for file in os.listdir(train_dir):
        images.append(file)
        path = train_dir + "/" + file
        arr = compress(path)
        if file != 'test.jpg':
            if len(Xtr) == 0:
                Xtr = arr
            else:
                Xtr = np.hstack((Xtr, arr))
        else:
            ytr = arr

    return Xtr, ytr


X_train, y_train = create_X_y()
X_train = np.vstack((X_train, np.ones((1, X_train.shape[1]))))

W = np.ones((X_train.shape[1], X_train.shape[0]))
W[:, 0:X_train.shape[0]] = np.ones((X_train.shape[1], X_train.shape[0])) / 10

y = {j.split('_1.jpg')[0]: i for i, j in enumerate(images) if j != 'test.jpg'}

def reg(W):
    squared = W ** 2
    L2 = np.sum(squared)

    return L2


def L_unvectorized(X, y, W):
    L = 0.0
    
    c = 1 / X.shape[1]
    delta = 1.0
    lambda = 1.0
    y_i = y['cat']

    for x_i in X.T:
        scores = W.dot(x_i)
        class_score = scores[y_i]

        m = 0.0
        for j in range(W.shape[0]):
            if j != y_i:
                m_i = np.maximum(0, scores[j] - class_score + delta)
                if m_i > 0:
                    m += m_i
        L += m

    L = L * c + (lambda * reg(W))

    return L

L_unvectorized(X_train, y, W)
