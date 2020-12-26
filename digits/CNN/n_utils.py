import numpy as np
MAX_EXP_ARG = 500

def init_params(im_shape, hparams):
    params = {}
    n_h, n_w, n_c = im_shape
    n_prev = n_c * n_h * n_w

    for l_name, l_hparams in hparams.items():
        params[l_name] = {}
        l_type = l_hparams["type"]

        if l_type == "conv":
            n_f, f_h, f_w, f_c = l_hparams["filters"]
            n_h, n_w, n_c = l_hparams["dimensions"]
            params[l_name]["W"] = np.random.randn(n_f, f_h, f_w, f_c)
            params[l_name]["b"] = np.zeros((n_f, 1))

            n_prev = n_h * n_w * n_c

        elif l_type == "fc" or l_type == "softmax":
            print("n_prev", n_prev)
            n_l = l_hparams["units"]
            params[l_name]["W"] = np.random.randn(n_l, n_prev) * np.sqrt(2./n_prev)
            params[l_name]["b"] = np.zeros((n_l, 1))
            n_prev = n_l

        elif l_type == "flatten":
            params[l_name]["W"] = np.array([])
            params[l_name]["b"] = np.array([])

    return params

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

    X_train = X[:, :m_train].reshape((28, 28, m_train, 1)).transpose((2, 0, 1, 3))
    Y_train = Y[:, :m_train]

    X_test = X[:, m_train :].reshape((28, 28, m_test, 1)).transpose((2, 0, 1, 3))
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

def apply_filter_slice(X_slice, W):
    return np.sum(X_slice * W)

def forward_FC_Step(A_prev, W, b, units):
    z = np.dot(W, A_prev) + b
    A = relu(z)
    return A

def forward_FC_Softmax(A_prev, W, b):
    z = np.dot(W, A_prev) + b
    A = softmax(z)
    return A

def forward_Conv_Step(A_prev, weights, biases, stride = (1,1)):
    m, n_H, n_W, _ = A_prev.shape
    n_f, fx, fy, _ = weights.shape
    sx, sy = stride
    n_H1 = int((n_H - fy) / sy + 1)
    n_W1 = int((n_W - fx) / sx + 1)

    res = np.zeros((m, n_H1, n_W1, n_f))

    for i in range(m):
        for f in range(n_f):
            W = weights[f]
            b = biases[f]
        for y in range(n_H1):
            for x in range(n_W1):
                Ai = A_prev[i]
                res[i][y][x][f] = apply_filter_slice(Ai[y * sy : y*sy + fy, x * sx : x * sx + fx], W) + b
    
    return relu(res)

def forward_full(X, params, hparams):
    A = X
    for l_name, l_params in params.items():
        l_hparams = hparams[l_name]
        l_type = l_hparams["type"]
        W = l_params["W"]
        b = l_params["b"]
        
        if (l_type == "conv"):
            s = l_hparams["strides"]
            A = forward_Conv_Step(A, W, b, s)
        elif l_type == "fc":
            units = l_hparams["units"]
            A = forward_FC_Step(A, W, b, units)
        elif l_type == "softmax":
            A = forward_FC_Softmax(A, W, b)
        elif l_type == "flatten":
            A = A.reshape((A.shape[0], -1)).T

    return A


def predict(X, params, hparams):
    a = forward_full(X, params, hparams)
    return (a == np.max(a, axis=0)) * 1

def saveTestData(params, h_layers):
    test_data = np.genfromtxt('digits/datasets/test.csv', delimiter=',')
    #test_data = np.genfromtxt('digits/datasets/test_debug.csv', delimiter=',')
    X_result = test_data[1:, :].T

    res = convert_from_one_hot(predict(X_result, h_layers, params)).reshape(X_result.shape[1], 1)
    submit_data = np.concatenate((np.arange(1, res.shape[0] + 1).reshape(X_result.shape[1], 1), res), axis=1)

    np.savetxt('digits/datasets/submition.csv', submit_data.astype(int), fmt='%i', delimiter=',', header='ImageId,Label')

#===========================================
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

hparams = {
    "conv-1": {
        "type": "conv",
        "strides": (1,1),
        "padding": (0,0),
        "filters": (16, 3, 3, 2),
        "dimensions": (26, 26, 16)
    },
    "flatten": {
        "type": "flatten"
    },
    "fc-1": {
        "type": "fc",
        "units": 256,
    },
    "fc-2": {
        "type": "fc",
        "units": 256,
    },
    "softmax": {
        "type": "softmax",
        "units": 10
    }
}
params = init_params(test_im.shape, hparams)
