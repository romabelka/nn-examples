#https://www.kaggle.com/c/digit-recognizer
import numpy as np
import n_utils
import matplotlib.pyplot as plt

n_labels = 10
MAX_EXP_ARG = 500
keep_prob = 0.8

data = np.genfromtxt('digits/datasets/train.csv', delimiter=',')

X = data.T[1:, 1:] / 255.
Y = n_utils.convert_to_one_hot(data.T[0:1, 1:].astype(int), n_labels)

m = X.shape[1]
m_test = 6000
m_dev = 0
m_train = m - m_test - m_dev

search_number = 5
print("total: ", m, ". Train: ", m_train, ". Dev: ", m_dev, ". Searching for: ", search_number)

X_train = X[:, :m_train]
Y_train = Y[:, :m_train]

X_dev = X[:, m_train : m_train + m_dev]
Y_dev = Y[:, m_train : m_train + m_dev]

X_test = X[:, m_train + m_dev :]
Y_test = Y[:, m_train + m_dev :]

n_x = X.shape[0]

print("m_train", m_train, ". m_test", m_test)

def softmax(z):
    e = np.exp(np.minimum(z, MAX_EXP_ARG))
    return e / np.sum(e, axis=0)

def relu(z):
    return np.fmax(0, z)

def init_params(n_x, h_layers, n_l):
    all_layers = [n_x] + h_layers + [n_l]
    params = {}
    for l in range(1, len(all_layers)):
        params["W" + str(l)] = np.random.randn(all_layers[l], all_layers[l-1]) * np.sqrt(2./all_layers[l-1])
        params["b" + str(l)] = np.zeros((all_layers[l], 1))
    return params

def forward_step(params, h_layers, X, keep_prob):
    A = X
    depth = len(h_layers) + 2
    cache = {"A0": A, "Z0": X, "D0": []}

    for l in range(1, depth):
        W = params["W" + str(l)]
        b = params["b" + str(l)]

        z = np.dot(W, A) + b

        D = (np.random.rand(z.shape[0], z.shape[1]) < keep_prob).astype(int)

        if (l == depth - 1):
            A = softmax(z)
        else:
            A = relu(z) * D / keep_prob

        cache["A" + str(l)] = A
        cache["Z" + str(l)] = z
        cache["D" + str(l)] = D


    return A, cache


def compute_cost(Y_hat, Y, params, lambd = 0):
    m = Y.shape[1]
    Y_hat[Y_hat == 0] = 1e-10
    Y_hat[Y_hat == 1] = 1 - 1e-10
    cost = -1/m * np.sum(Y * np.log(Y_hat))

    depth = len(params) // 2
    reg = 0
    for l in range(1, depth + 1):
        reg += np.linalg.norm(params["W" + str(l)]) ** 2 * lambd / (2 * m)

    return cost + reg

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0
        
    return dZ 

def get_grads(cache, X, Y, params, lambd, keep_prob):
    m = X.shape[1]
    L = len(cache) // 3 - 1
    grads = {}
    A = cache["A" + str(L)]
    dZ = A - Y # (n_l,m)

    for l in range(L, 0, -1):
        A_prev = cache["A" + str(l - 1)]
        W = params["W" + str(l)]

        dw = np.dot(dZ, A_prev.T) / m + W * lambd / m # (n_l, m) * (m, n) = (n_l, n)
        db = np.sum(dZ, axis=1, keepdims=True) / m
            
        grads["dW" + str(l)] = dw 
        grads["db" + str(l)] = db

        if (l > 1):
            D_prev = cache["D" + str(l - 1)]
            dA = np.dot(W.T, dZ) * D_prev / keep_prob
            dZ = relu_backward(dA, cache["Z" + str(l - 1)])

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

def show_correctness(params, h_layers):
        train_correctness = get_correctness(predict(X_train, h_layers, params), Y_train)
        test_correctness = get_correctness(predict(X_test, h_layers, params), Y_test)
        print("TRAIN. Correctness: ", train_correctness, " %")
        print("TEST. Correctness: ", test_correctness, " %")


def model_adam(X, Y, num_epochs, lambd, keep_prob, learning_rate, h_layers, minibatch_size):
    params = init_params(X.shape[0], h_layers, n_labels)
    minibatches = n_utils.get_minibatches(X, Y, minibatch_size)
    l_dacay_rate = 0.01
    t = 0
    v, s = init_adam(params)
    costs = []
#    best_params = {}
#    best_test_perf = 0

    for i in range(0, num_epochs + 1):
        cost_total = 0

        for minibatch in minibatches:
            t += 1
            X_mini, Y_mini = minibatch
            Y_hat, cache = forward_step(params, h_layers, X_mini, keep_prob = keep_prob)
            cost_total += compute_cost(Y_hat, Y_mini, params, lambd)
            grads = get_grads(cache, X_mini, Y_mini, params, lambd, keep_prob = keep_prob)
            params, v, s = update_params_adam(params, grads, v, s, t, learning_rate / (1 + l_dacay_rate * i), beta1 = 0.8)

#        test_correctness = get_correctness(predict(X_test, h_layers, params), Y_test)
#        if (test_correctness > best_test_perf):
#            best_test_perf = test_correctness
#            best_params = params

        costs.append(cost_total)

        if (i % 50 == 0):
            show_correctness(params, h_layers)
            print("cost after " + str(i) + ": ", cost_total, ". Learning rate: ", learning_rate / (1 + l_dacay_rate * i))

    return params, costs


def model(X, Y, num_epochs = 500, lambd = 0.1, learning_rate=0.1, h_layers=[2, 5, 4], minibatch_size=128, keep_prob=0.8):
    params = init_params(X.shape[0], h_layers, n_labels)
    minibatches = n_utils.get_minibatches(X, Y, minibatch_size)
    l_dacay_rate = 0.001
    costs = []

    for i in range(0, num_epochs + 1):
        cost_total = 0

        for minibatch in minibatches:
            X_mini, Y_mini = minibatch
            Y_hat, cache = forward_step(params, h_layers, X_mini, keep_prob = keep_prob)
            cost_total += compute_cost(Y_hat, Y_mini, params, lambd)
            grads = get_grads(cache, X_mini, Y_mini, params, lambd, keep_prob = keep_prob)
            params = update_params(params, grads, learning_rate / (1 + l_dacay_rate * i))
            if (np.isnan(cost_total)):
                print("Nan after iteration: ", i, grads, params)
                return params

        costs.append(cost_total)

        if (i % 100 == 0):
            print("cost after " + str(i) + ": ", cost_total, ". Learning rate: ", learning_rate / (1 + l_dacay_rate * i))

    return params, costs

def predict(X, h_layers, params):
    a, _ = forward_step(params, h_layers, X, keep_prob = 1)
    return (a == np.max(a, axis=0)) * 1

def get_correctness(predict, Y):
    total = Y.shape[1]
    predict_res = n_utils.convert_from_one_hot(predict)
    truth = n_utils.convert_from_one_hot(Y)

    acuracy = 100 * np.sum(np.equal(predict_res, truth)) / total
    return acuracy

def plot_cost(costs):
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.show()

def tune_hyper_params(architectures = [[10], [20], [10, 5], [20, 5]], lambds = [0], keep_probs= [0], iterations = 1000, learning_rate = 0.001):
    for i in range(0, len(architectures)):
        h_layers = architectures[i]
        for j in range(0, len(lambds)):
            for keep_prob in keep_probs:
                lambd = lambds[j]

                params, costs = model_adam(X_train, Y_train, num_epochs = iterations, minibatch_size = 128, learning_rate = learning_rate, lambd = lambd, keep_prob = keep_prob, h_layers = h_layers)            
                #params = model(X_train, Y_train, num_epochs = iterations, minibatch_size = 128, learning_rate = learning_rate, lambd = lambd, h_layers = h_layers)
                train_correctness = get_correctness(predict(X_train, h_layers, params), Y_train)
                test_correctness = get_correctness(predict(X_test, h_layers, params), Y_test)

                plot_cost(costs)

                print("Deep NN. Hidden Layers: ", h_layers, ". Regularization lambda = ", lambd, ". # of iterations: ", iterations)
                print("TRAIN. Correctness: ", train_correctness, " %")
                print("TEST. Correctness: ", test_correctness, " %")
                print("===============================")
            
h_layers = [60, 40, 30, 30]

#tune_hyper_params(architectures=[h_layers], lambds=[0], keep_probs = [.8], iterations=300, learning_rate=0.004)

params, costs = model_adam(X_train, Y_train, num_epochs = 1000, minibatch_size = 128, learning_rate = 0.004, lambd = 0, keep_prob=0.8, h_layers = h_layers)            

def imshow(X, i):
    plt.imshow(X[:, i].reshape(28,28))
    plt.show()

def save_test_data(params, h_layers):
    test_data = np.genfromtxt('digits/datasets/test.csv', delimiter=',')
    #test_data = np.genfromtxt('digits/datasets/test_debug.csv', delimiter=',')
    X_result = test_data[1:, :].T

    res = n_utils.convert_from_one_hot(predict(X_result, h_layers, params)).reshape(X_result.shape[1], 1)
    submit_data = np.concatenate((np.arange(1, res.shape[0] + 1).reshape(X_result.shape[1], 1), res), axis=1)

    np.savetxt('digits/datasets/submition.csv', submit_data.astype(int), fmt='%i', delimiter=',', header='ImageId,Label')

save_test_data(params, h_layers)
plot_cost(costs)
#============================================
#Deep NN. Hidden Layers:  [30, 20, 15, 10, 10, 5] . Regularization lambda =  0 . # of iterations:  200
#TRAIN. Correctness:  99.04722222222222  %
#TEST. Correctness:  95.11666666666666  %

#cost after 50:  34.14852149414704 . Learning rate:  0.0026666666666666666
#TRAIN. Correctness:  99.33333333333333  %
#TEST. Correctness:  96.43333333333334  %

#TRAIN. Correctness:  99.88055555555556  %
#TEST. Correctness:  95.36666666666666  %
#cost after 1000:  10.06735110434442 . Learning rate:  0.00036363636363636367