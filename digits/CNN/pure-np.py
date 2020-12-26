import numpy as np
import n_utils
from n_utils import softmax, relu, forward_full, predict, init_params
import matplotlib.pyplot as plt

X_train, Y_train, X_test, Y_test, n_x, m, m_test, m_train, n_labels = n_utils.load_data(dev=True)
print("m_train", m_train, ". m_test", m_test, X_train.shape)

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

#params = init_params((28, 28, 1), hparams)

def compute_cost(Y_hat, Y, params, lambd = 0):
    m = Y.shape[1]
    Y_hat[Y_hat == 0] = 1e-10
    Y_hat[Y_hat == 1] = 1 - 1e-10
    cost = -1/m * np.sum(Y * np.log(Y_hat))

    #depth = len(params)
    reg = 0
    #for l in range(1, depth + 1):
    #    reg += np.linalg.norm(params["W" + str(l)]) ** 2 * lambd / (2 * m)

    return cost + reg

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0
        
    return dZ 

def back_softmax(A_prev, A, Y, W, b):
    m = A.shape[1]
    dZ = A - Y
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m

    dA = np.dot(W.T, dZ)
    
    return dW, db, dA

def back_FC(A_prev, Z, dA, W, b):
    m = A_prev.shape[1]
    dZ = relu_backward(dA, Z)
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
        
    dA = np.dot(W.T, dZ)
    return dW, db, dA

#TODO
def back_flatten(W):
    dW = np.array([])
    db = np.array([])
    dA = np.array([])
    return dW, db, dA

def back_conv(W, b):
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)
    dA = np.array([])

    return dW, db, dA


def compute_grads(cache, Y, params, hparams):
    grads = {}
    dW = db = dA = np.array([])

    for l_name, l_hparams in reversed(hparams.items()):
        l_type = l_hparams["type"]
        W = params[l_name]["W"]
        b = params[l_name]["b"]
        A = cache[l_name]["A"]
        Z = cache[l_name]["Z"]
        A_prev = cache[l_name]["A_prev"]

        if l_type == "softmax":
            dW, db, dA = back_softmax(A_prev, A, Y, W, b)
        elif l_type == "fc":
            dW, db, dA = back_FC(A_prev, Z, dA, W, b)
        elif l_type == "flatten":
            dW, db, dA = back_flatten(W)
        elif l_type == "conv":
            dW, db, dA = back_conv(W, b)
        
        grads[l_name] = {
            "dW": dW,
            "db": db
        }

    return grads


def update_params(params, grads, learning_rate):
    for l_name, l_grads in grads.items():
        params[l_name]["W"] -= learning_rate * l_grads["dW"]
        params[l_name]["b"] -= learning_rate * l_grads["db"]
    
    return params

def init_adam(params):
    L = len(params) // 2
    v = {}
    s = {}

    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros(params["W" + str(l)].shape)
        v["db" + str(l)] = np.zeros(params["b" + str(l)].shape)
        s["dW" + str(l)] = np.zeros(params["W" + str(l)].shape)
        s["db" + str(l)] = np.zeros(params["b" + str(l)].shape)
    
    return v, s


def update_params_adam(params, grads, v, s, t, learning_rate, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-6):
    L = len(params) // 2
    #v - moving avarage grads
    #s - moving avarage square grads
    for l in range(1, L + 1):
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]

        vw_corrected = v["dW" + str(l)] / (1 - beta1 ** t)
        vb_corrected = v["db" + str(l)] / (1 - beta1 ** t)

        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * np.power(grads["dW" + str(l)], 2)
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * np.power(grads["db" + str(l)], 2)

        sw_corrected = s["dW" + str(l)] / (1 - beta2 ** t)
        sb_corrected = s["db" + str(l)] / (1 - beta2 ** t)

        params["W" + str(l)] = params["W" + str(l)] - learning_rate * vw_corrected / (np.sqrt(sw_corrected) + epsilon)
        params["b" + str(l)] = params["b" + str(l)] - learning_rate * vb_corrected / (np.sqrt(sb_corrected) + epsilon)

    return params, v, s

def show_correctness(params, h_layers):
        train_correctness = get_correctness(predict(X_train, h_layers, params), Y_train)
        test_correctness = get_correctness(predict(X_test, h_layers, params), Y_test)
        print("TRAIN. Correctness: ", train_correctness, " %")
        print("TEST. Correctness: ", test_correctness, " %")


def model_adam(X, Y, num_epochs, lambd, learning_rate, h_layers, minibatch_size):
    params = init_params(X.shape[0], h_layers, n_labels)
    minibatches = n_utils.get_minibatches(X, Y, minibatch_size)
    l_dacay_rate = 0.01
    t = 0
    v, s = init_adam(params)

    for i in range(0, num_epochs + 1):
        cost_total = 0

        for minibatch in minibatches:
            t += 1
            X_mini, Y_mini = minibatch
            Y_hat, cache = forward_FC(params, h_layers, X_mini)
            cost_total += compute_cost(Y_hat, Y_mini, params, lambd)
            grads = grad_descend(cache, X_mini, Y_mini, params, lambd)
            params, v, s = update_params_adam(params, grads, v, s, t, learning_rate / (1 + l_dacay_rate * i), beta1 = 0.8)

        if (i % 50 == 0):
            show_correctness(params, h_layers)
            print("cost after " + str(i) + ": ", cost_total, ". Learning rate: ", learning_rate / (1 + l_dacay_rate * i))

    return params


def model(X, Y, hparams, num_epochs = 500, lambd = 0.1, learning_rate=0.1, minibatch_size=64):
    params = init_params((28, 28, 1), hparams)
    minibatches = n_utils.get_minibatches(X, Y, minibatch_size)
    l_dacay_rate = 0.001

    for i in range(0, num_epochs + 1):
        cost_total = 0

        for minibatch in minibatches:
            X_mini, Y_mini = minibatch
            Y_hat, cache = forward_full(X_mini, params, hparams)
            cost_total += compute_cost(Y_hat, Y_mini, params, lambd)
            grads = compute_grads(cache, Y_mini, params, hparams)
            params = update_params(params, grads, learning_rate / (1 + l_dacay_rate * i))
 
        if (i % 1 == 0):
            print("cost after " + str(i) + ": ", cost_total, ". Learning rate: ", learning_rate / (1 + l_dacay_rate * i))

    return params

def get_correctness(predict, Y):
    total = Y.shape[1]
    predict_res = n_utils.convert_from_one_hot(predict)
    truth = n_utils.convert_from_one_hot(Y)

    acuracy = 100 * np.sum(np.equal(predict_res, truth)) / total
    return acuracy

def tune_hyper_params(lambds = [0, 0.1, 1], iterations = 1000, learning_rate = 0.001):
    for j in range(0, len(lambds)):
        lambd = lambds[j]

        #params = model_adam(X_train, Y_train, num_epochs = iterations, minibatch_size = 128, learning_rate = learning_rate, lambd = lambd, h_layers = h_layers)            
        params = model(X_train, Y_train, hparams, num_epochs = iterations, minibatch_size = 64, learning_rate = learning_rate, lambd = lambd)
        train_correctness = get_correctness(predict(X_train, params, hparams), Y_train)
        test_correctness = get_correctness(predict(X_test, params, hparams), Y_test)

        print("TRAIN. Correctness: ", train_correctness, " %")
        print("TEST. Correctness: ", test_correctness, " %")
            
tune_hyper_params(lambds=[0], iterations=5, learning_rate=0.001)

#params = model_adam(X_train, Y_train, num_epochs = 100, minibatch_size = 64, learning_rate = 0.001, lambd = 0.7, h_layers = h_layers)            

def imshow(X, i):
    plt.imshow(X[i])
    plt.show()


#n_utils.saveTestData(params, h_layers)
