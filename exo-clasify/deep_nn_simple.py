#note: Get back here when I learn how to deal with data when 99% zeros and 1% ones
#https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data
import numpy as np

test_data = np.genfromtxt('exo-clasify/dataset/exo_test.csv', delimiter=',')
train_data = np.genfromtxt('exo-clasify/dataset/exo_train.csv', delimiter=',')

m_train = train_data.shape[0] - 1
m_test = test_data.shape[0] - 1
n_x = test_data.shape[1] - 1

Y_test = (test_data[1:, 0] - 1).reshape(1, m_test)
X_test = (test_data[1:, 1:]).reshape(n_x, m_test) / 10000

Y_train = (train_data[1:, 0] - 1).reshape(1, m_train)
X_train = (train_data[1:, 1:]).reshape(n_x, m_train) / 10000

print(Y_test.shape)

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

def model(X, Y, iterations = 1000, lambd = 0.1, learning_rate=0.1, h_layers=[2, 5, 4], print_cost=False):
    params = init_params(X.shape[0], h_layers)

    for i in range(0, iterations + 1):
        Y_hat, cache = forward_step(params, h_layers, X)
        cost = compute_cost(Y_hat, Y, params, lambd)
        grads = grad_descend(cache, X, Y, params, lambd)
        params = update_params(params, grads, learning_rate)
        if (print_cost and i % 100 == 0):
            print("cost after " + str(i) + ": ", cost)

    return params

def predict(X, h_layers, params):
    a, _ = forward_step(params, h_layers, X)
    return (a > 0.5) * 1

def get_correctness(predict, Y):
    total = Y.shape[1]
    positives = np.sum(Y)
    negatives = total - positives
    print (positives, negatives)

    false_positive = np.sum(np.logical_and(predict, np.logical_not(Y))) 
    false_negative = np.sum(np.logical_and(Y, np.logical_not(predict)))

    return false_negative, false_positive

def tune_hyper_params(architectures = [[10], [20], [10, 5], [20, 5]], lambds = [0, 0.1, 1], iterations = 5000, learning_rate = 0.01):
    for i in range(0, len(architectures)):
        h_layers = architectures[i]
        for j in range(0, len(lambds)):
            lambd = lambds[j]

            params = model(X_train, Y_train, iterations = iterations, learning_rate = learning_rate, lambd = lambd, h_layers = h_layers, print_cost=True)
            train_correctness = get_correctness(predict(X_train, h_layers, params), Y_train)
            test_correctness = get_correctness(predict(X_test, h_layers, params), Y_test)

            print("Deep NN. Hidden Layers: ", h_layers, ". Regularization lambda = ", lambd, ". # of iterations: ", iterations)
            print("TRAIN. False Negative: ", train_correctness[0], ". Flase Positive: ", train_correctness[1])
            print("TEST. False Negative: ", test_correctness[0], ". Flase Positive: ", test_correctness[1])
            
tune_hyper_params(architectures=[[], [20], [20, 10], [20, 10, 5]], lambds=[0], iterations=1000, learning_rate=0.01)

#Deep NN. Hidden Layers:  [20, 10, 5] . Regularization lambda =  0 . # of iterations:  1000
#train correctness:  99.27265578926676 %
#test correctness:  99.12280701754386 %

#Deep NN. Hidden Layers:  [] . Regularization lambda =  0 . # of iterations:  100
#TRAIN. False Negative:  11 . Flase Positive:  65
#TEST. False Negative:  0 . Flase Positive:  452

#=====Best Result===============
#Deep NN. Hidden Layers:  [20] . Regularization lambda =  0 . # of iterations:  1000
#TRAIN. False Negative:  1, of 5050. Flase Positive:  1, of 37
#TEST. False Negative:  5, of 565 . Flase Positive:  1, 5