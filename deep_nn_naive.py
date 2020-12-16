import numpy as np
import improv_utils
import matplotlib.pyplot as plt

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = improv_utils.load_cats()

def convert_y_labels(Y):
    return np.array((Y == 0) * 1)

def reshape_X(X):
    return np.reshape(X, (X.shape[0], -1)).T / 255.

m_train = X_train_orig.shape[0]
m_test = X_test_orig.shape[0]
print("m_train", m_train, ". m_test", m_test)

X_train = reshape_X(X_train_orig)
X_test = reshape_X(X_test_orig)
n_x = X_train.shape[0]

Y_train = Y_train_orig
Y_test = Y_test_orig

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.fmax(0, z)

def init_params(n, h_layers):
    all_layers = [n] + h_layers + [1]
    params = {}
    for l in range(1, len(all_layers)):
        params["W" + str(l)] = np.random.randn(all_layers[l], all_layers[l-1]) * 0.01
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

#A, cache = forward_step(init_params(n_x, h_layers), h_layers, X_test)
#print('cost', compute_cost(A, Y_test, init_params(n_x, h_layers), 100))
    
def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
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

def update_params(params, grads, learning_rate = 0.001):
    for l in range(1, len(params) // 2 + 1):
        params["W" + str(l)] = params["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        params["b" + str(l)] = params["b" + str(l)] - learning_rate * grads["db" + str(l)]
    
    return params

def model(X, Y, iterations = 1000, lambd = 0.1, learning_rate=0.1, h_layers=[2, 5, 4]):
    params = init_params(X.shape[0], h_layers)

    for i in range(0, iterations):
        Y_hat, cache = forward_step(params, h_layers, X)
        cost = compute_cost(Y_hat, Y, params, lambd)
        grads = grad_descend(cache, X, Y, params, lambd)
        params = update_params(params, grads, learning_rate)
        if (i % 100 == 0):
            print("cost after " + str(i) + ": ", cost)

    return params

def predict(X, h_layers, params):
    a, _ = forward_step(params, h_layers, X)
    return (a > 0.5) * 1

def get_correctness(predict, Y):
    total = Y.shape[1]
    wrong = np.sum(np.logical_xor(predict, Y))
    return (1 - wrong / total) * 100

h_layers = [2, 7, 5]
#params = model(X_train, Y_train, iterations=10, learning_rate = 1, lambd = 0, h_layers = h_layers)
#theta, shapes = improv_utils.dict_to_vector(params)
#restored = improv_utils.vector_to_dict(theta, shapes)
#print(np.equal(restored["W1"], params["W1"]))
#print(np.equal(restored["b2"], params["b2"]))

def dubug_grad_descend(X, Y, epsilon = 0.000001, h_layers = h_layers, learning_rate = 0.1, lambd=0):
    params = model(X, Y, iterations=2, learning_rate = learning_rate, lambd = lambd, h_layers = h_layers)
    theta, shapes = improv_utils.dict_to_vector(params)

    A, cache = forward_step(params, h_layers, X)
    grads, _ = improv_utils.dict_to_vector(grad_descend(cache, X, Y, params, lambd))
    grad_approx = np.zeros(theta.shape)

    theta, shapes = improv_utils.dict_to_vector(params)

    print(theta.shape, shapes)
    
    for i in range(theta.shape[0]):
        theta_plus = np.copy(theta)
        theta_plus[i][0] = theta_plus[i][0] + epsilon 
    
        theta_minus = np.copy(theta)
        theta_minus[i][0] = theta_minus[i][0] - epsilon 

        A_plus, _ = forward_step(improv_utils.vector_to_dict(theta_plus, shapes), h_layers, X)
        A_minus, _ = forward_step(improv_utils.vector_to_dict(theta_minus, shapes), h_layers, X)
        
        cost_plus = compute_cost(A_plus, Y, improv_utils.vector_to_dict(theta_plus, shapes), lambd)
        cost_minus = compute_cost(A_minus, Y, improv_utils.vector_to_dict(theta_minus, shapes), lambd)

        grad_approx[i] = (cost_plus - cost_minus) / (2*epsilon)
        if (i % 1000 == 0):
            print(i, " iterations")

    numerator = np.linalg.norm(grad_approx - grads)                                           
    denominator = np.linalg.norm(grads) + np.linalg.norm(grad_approx)

    return numerator / denominator

print("grad Difference is:", dubug_grad_descend(X_train, Y_train))    



#train_correctness = get_correctness(predict(X_train, h_layers, params), Y_train)
#test_correctness = get_correctness(predict(X_test, h_layers, params), Y_test)

#print("Deep NN without regularization")
#print("train correctness: ", train_correctness, "%")
#print("test correctness: ", test_correctness, "%")

#Cats vs noncats after 10K iterations, lambda = 1
#train correctness:  65.55023923444976 %
#test correctness:  34.0 %
