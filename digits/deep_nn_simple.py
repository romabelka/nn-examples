#https://www.kaggle.com/c/digit-recognizer
import numpy as np
import n_utils

data = np.genfromtxt('digits/datasets/train.csv', delimiter=',')
X = data.T[1:, 1:] / 255.
Y = data.T[0:1, 1:]
m = X.shape[1]
m_test = 2000
m_dev = 2000
m_train = m - m_test - m_dev

search_number = 5
print("total: ", m, ". Train: ", m_train, ". Dev: ", m_dev, ". Searching for: ", search_number)

X_train = X[:, :m_train]
Y_train = Y[:, :m_train]
Y_train = (Y_train == search_number) * 1 # Just to start with binary regression

X_dev = X[:, m_train : m_train + m_dev]
Y_dev = Y[:, m_train : m_train + m_dev]
Y_dev = (Y_dev == search_number) * 1 # Just to start with binary regression

X_test = X[:, m_train + m_dev :]
Y_test = Y[:, m_train + m_dev :]
Y_test = (Y_test == search_number) * 1 # Just to start with binary regression

n_x = X.shape[0]

print(data.shape)

print("m_train", m_train, ". m_test", m_test)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.fmax(0, z)

def init_params(n, h_layers):
    all_layers = [n] + h_layers + [1]
    params = {}
    for l in range(1, len(all_layers)):
        params["W" + str(l)] = np.random.randn(all_layers[l], all_layers[l-1]) * np.sqrt(2./all_layers[l-1])
        params["b" + str(l)] = np.zeros((all_layers[l], 1))
    return params

def forward_step(params, h_layers, X):
    A = X
    depth = len(h_layers) + 2
    cache = {"A0": A, "Z0": X}

    for l in range(1, depth):
        W = params["W" + str(l)]
        b = params["b" + str(l)]

        z = np.dot(W, A) + b
        if (l == depth - 1):
            A = sigmoid(z)
        else:
            A = relu(z)

        cache["A" + str(l)] = A
        cache["Z" + str(l)] = z


    return A, cache


def compute_cost(Y_hat, Y, params, lambd = 0):
    m = Y.shape[1]
    Y_hat[Y_hat == 0] = 1e-10
    Y_hat[Y_hat == 1] = 1 - 1e-10
    cost = np.squeeze(-np.dot(Y, np.log(Y_hat).T) - np.dot((1 - Y), np.log(1 - Y_hat).T))

    depth = len(params) // 2
    reg = 0
    for l in range(1, depth + 1):
        reg += np.linalg.norm(params["W" + str(l)]) ** 2 * lambd / (2 * m)

    return cost + reg

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0
        
    return dZ 

def grad_descend(cache, X, Y, params, lambd):
    m = X.shape[1]
    L = len(cache) // 2 - 1
    grads = {}
    A = cache["A" + str(L)]
    dz = A - Y # (1,m)

    for l in range(L, 0, -1):
        A_prev = cache["A" + str(l - 1)]
        W = params["W" + str(l)]
        dw = np.dot(dz, A_prev.T) / m + W * lambd / m # (1, m) * (m, n) = (1, n)
        db = np.sum(dz, axis=1, keepdims=True) / m
        dA = np.dot(W.T, dz)
        grads["dW" + str(l)] = dw 
        grads["db" + str(l)] = db
        dz = relu_backward(dA, cache["Z" + str(l - 1)])

    return grads

def update_params(params, grads, learning_rate):
    for l in range(1, len(params) // 2 + 1):
        params["W" + str(l)] = params["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        params["b" + str(l)] = params["b" + str(l)] - learning_rate * grads["db" + str(l)]
    
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

def model_adam(X, Y, num_epochs, lambd, learning_rate, h_layers, minibatch_size):
    params = init_params(X.shape[0], h_layers)
    minibatches = n_utils.get_minibatches(X, Y, minibatch_size)
    t = 0
    v, s = init_adam(params)

    for i in range(0, num_epochs + 1):
        cost_total = 0

        for minibatch in minibatches:
            t += 1
            X_mini, Y_mini = minibatch
            Y_hat, cache = forward_step(params, h_layers, X_mini)
            cost_total += compute_cost(Y_hat, Y_mini, params, lambd)
            grads = grad_descend(cache, X_mini, Y_mini, params, lambd)
            params, v, s = update_params_adam(params, grads, v, s, t, learning_rate, beta1 = 0.8)

        if (i % 50 == 0):
            print("cost after " + str(i) + ": ", cost_total)

    return params


def model(X, Y, num_epochs = 500, lambd = 0.1, learning_rate=0.1, h_layers=[2, 5, 4], minibatch_size=128):
    params = init_params(X.shape[0], h_layers)
    minibatches = n_utils.get_minibatches(X, Y, minibatch_size)

    for i in range(0, num_epochs + 1):
        cost_total = 0

        for minibatch in minibatches:
            X_mini, Y_mini = minibatch
            Y_hat, cache = forward_step(params, h_layers, X_mini)
            cost_total += compute_cost(Y_hat, Y_mini, params, lambd)
            grads = grad_descend(cache, X_mini, Y_mini, params, lambd)
            params = update_params(params, grads, learning_rate)

        if (i % 50 == 0):
            print("cost after " + str(i) + ": ", cost_total)

    return params

def predict(X, h_layers, params):
    a, _ = forward_step(params, h_layers, X)
    return (a > 0.5) * 1

def get_correctness(predict, Y):
    total = Y.shape[1]
    wrong = np.sum(np.logical_xor(predict, Y))
    return (1 - wrong / total) * 100

def tune_hyper_params(architectures = [[10], [20], [10, 5], [20, 5]], lambds = [0, 0.1, 1], iterations = 1000, learning_rate = 0.01):
    for i in range(0, len(architectures)):
        h_layers = architectures[i]
        for j in range(0, len(lambds)):
            lambd = lambds[j]

            params = model_adam(X_train, Y_train, num_epochs = iterations, minibatch_size = 128, learning_rate = learning_rate, lambd = lambd, h_layers = h_layers)
            train_correctness = get_correctness(predict(X_train, h_layers, params), Y_train)
            test_correctness = get_correctness(predict(X_test, h_layers, params), Y_test)

            print("Deep NN. Hidden Layers: ", h_layers, ". Regularization lambda = ", lambd, ". # of iterations: ", iterations)
            print("TRAIN. Correctness: ", train_correctness, " %")
            print("TEST. Correctness: ", test_correctness, " %")
            
tune_hyper_params(architectures=[[20, 10, 10, 5]], lambds=[0], iterations=250, learning_rate=0.01)

#--------------Adam---------------------
# Looks like Adam doesn't work well with regularization

#Deep NN. Hidden Layers:  [20, 10, 5] . Regularization lambda =  0 . # of iterations:  600
#TRAIN. Correctness:  99.99736842105264  %
#TEST. Correctness:  99.4  %

#Deep NN. Hidden Layers:  [20, 10, 10, 5] . Regularization lambda =  0 . # of iterations:  300
#TRAIN. Correctness:  100.0  %
#TEST. Correctness:  99.4  %

#--------------Mini Batch---------------
#Deep NN. Hidden Layers:  [20] . Regularization lambda =  0 . # of iterations:  200, batch_size = 128
#TRAIN. Correctness:  99.52368421052633  %
#TEST. Correctness:  98.95  %

#Deep NN. Hidden Layers:  [20, 10, 5] . Regularization lambda =  0 . # of iterations:  400, batch_size = 128
#TRAIN. Correctness:  99.9921052631579  %
#TEST. Correctness:  99.05000000000001  %

#Deep NN. Hidden Layers:  [20, 10, 5] . Regularization lambda =  0.5 . # of iterations:  600
#TRAIN. Correctness:  99.79736842105264  %
#TEST. Correctness:  99.2  %

#--------------Batch---------------------
#Deep NN. Hidden Layers:  [20] . Regularization lambda =  0 . # of iterations:  1000
#TRAIN. Correctness:  94.87105263157895  %
#TEST. Correctness:  95.3  %