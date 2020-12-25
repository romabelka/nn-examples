import numpy as np
MAX_EXP_ARG = 500

def shuffle_set(X, Y):
    m = X.shape[1]
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

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

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def convert_from_one_hot(Y):
    return np.argmax(Y, axis=0)

def load_data(dev = False, test_percent = 0.2):
    n_labels = 10
    data = []
    if dev:
        data = np.genfromtxt('digits/datasets/train_debug.csv', delimiter=',')
    else:
        data = np.genfromtxt('digits/datasets/train.csv', delimiter=',')

    X = data.T[1:, 1:] / 255.
    Y = convert_to_one_hot(data.T[0:1, 1:].astype(int), n_labels)

    m = X.shape[1]
    m_test = int(m * test_percent)
    m_train = m - m_test

    print("total: ", m, ". Train: ", m_train, ". Test: ", m_test)

    X_train = X[:, :m_train]
    Y_train = Y[:, :m_train]

    X_test = X[:, m_train :]
    Y_test = Y[:, m_train :]

    n_x = X.shape[0]
    
    return X_train, Y_train, X_test, Y_test, n_x, m, m_test, m_train, n_labels

def softmax(z):
    e = np.exp(np.minimum(z, MAX_EXP_ARG))
    return e / np.sum(e, axis=0)

def relu(z):
    return np.fmax(0, z)

def forward_FC(params, h_layers, X):
    A = X
    depth = len(h_layers) + 2
    cache = {"A0": A, "Z0": X}

    for l in range(1, depth):
        W = params["W" + str(l)]
        b = params["b" + str(l)]

        z = np.dot(W, A) + b
        if (l == depth - 1):
            A = softmax(z)
        else:
            A = relu(z)

        cache["A" + str(l)] = A
        cache["Z" + str(l)] = z


    return A, cache

def predict(X, h_layers, params):
    a, _ = forward_FC(params, h_layers, X)
    return (a == np.max(a, axis=0)) * 1

def saveTestData(params, h_layers):
    test_data = np.genfromtxt('digits/datasets/test.csv', delimiter=',')
    #test_data = np.genfromtxt('digits/datasets/test_debug.csv', delimiter=',')
    X_result = test_data[1:, :].T

    res = convert_from_one_hot(predict(X_result, h_layers, params)).reshape(X_result.shape[1], 1)
    submit_data = np.concatenate((np.arange(1, res.shape[0] + 1).reshape(X_result.shape[1], 1), res), axis=1)

    np.savetxt('digits/datasets/submition.csv', submit_data.astype(int), fmt='%i', delimiter=',', header='ImageId,Label')

def apply_filter_slice(X_slice, W):
    return np.sum(X_slice * W)

def conv_step(X, weights, biases, stride = (1,1)):
    m, n_H, n_W, _ = X.shape
    n_f, fx, fy, _ = weights.shape
    sx, sy = stride
    n_H1 = int((n_H - fy) / sy + 1)
    n_W1 = int((n_W - fx) / sx + 1)

    res = np.zeros((m, n_H1, n_W1, n_f))

    for i in range(X.shape[0]):
        for f in range(n_f):
            W = weights[f]
            b = biases[f]
        for y in range(n_H1):
            for x in range(n_W1):
                res[i][y][x][f] = apply_filter_slice(X[i][y * sy : y*sy + fy, x * sx : x * sx + fx], W) + b
    
    return res
        

test_im = np.array([
    [1, 2, 3, 50, 10, 0],
    [1, 2, 3, 60, 20, 0],
    [0, 0, 10, 50, 10, 0],
    [0, 0, 10, 50, 10, 0],
])

test_im = np.stack((test_im, test_im * 2), axis=2)

test_W = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])
test_W = np.stack((test_W, test_W * 2), axis=2)

test_X = np.stack((test_im, test_im, test_im, test_im, test_im), axis=0)

weights = np.stack((test_W, test_W, test_W),)
biases = np.stack(([0], [1], [2]))

print(test_X.shape)
print(conv_step(test_X, weights, biases, stride=(2,2)).shape)