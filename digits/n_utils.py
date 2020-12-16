import numpy as np

def shuffle_set(X, Y):
    m = X.shape[1]
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    return shuffled_X, shuffled_Y

def get_minibatches(X, Y, size):
    m = X.shape[1]
    shuffled_X, shuffled_Y = shuffle_set(X, Y)

    minibatches = []
    full_minibatches = m // size
    
    for i in range(0, full_minibatches):
        X_minibatch = shuffled_X[:, i * size : (i + 1) * size]
        Y_minibatch = shuffled_Y[:, i * size : (i + 1) * size]
        minibatches.append((X_minibatch, Y_minibatch))

    if (full_minibatches * size != m):
        X_minibatch = X[:, full_minibatches * size : ]
        Y_minibatch = Y[:, full_minibatches * size : ]
        minibatches.append((X_minibatch, Y_minibatch))

    return minibatches