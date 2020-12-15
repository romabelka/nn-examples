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

Y_train = Y_train_orig
Y_test = Y_test_orig

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def init_params(n):
    params = {}
    params["W1"] = np.random.randn(1, n) * 0.01
    params["b1"] = np.zeros((1, 1))
    return params

def forward_step(params, X):
    W1 = params["W1"]
    b = params["b1"]
    z = np.dot(W1, X) + b
    a = sigmoid(z)

    return a

def compute_cost(Y_hat, Y, W, lambd):
    m = Y.shape[1]
    return np.squeeze(-np.dot(Y, np.log(Y_hat).T) - np.dot((1 - Y), np.log(1 - Y_hat).T)) + np.linalg.norm(W) ** 2 * lambd / (2 * m)

def grad_descend(a, x, y,  w, lambd):
    m = x.shape[1]
    
    #da = -y/a + (1-y)/(1-a) # (1,m)
    dz = a - y # (1,m)
    
    #dw = np.dot(x, dz.T) / m # (n, m) * (m, 1) = (n, 1)
    dw = np.dot(dz, x.T) / m + w * lambd / m # (1, m) * (m, n) = (1, n)
    db = np.sum(dz) / m

    return {
        "dW1": dw,
        "db1": db
    }

def update_params(params, grads, learning_rate = 0.001):
    params["W1"] = params["W1"] - learning_rate * grads["dW1"]
    params["b1"] = params["b1"] - learning_rate * grads["db1"]
    return params

def model(X, Y, iterations = 1000, lambd = 0.1):
    params = init_params(X.shape[0])

    for i in range(0, iterations):
        Y_hat = forward_step(params, X)
        cost = compute_cost(Y_hat, Y, params["W1"], lambd)
        grads = grad_descend(Y_hat, X, Y, params["W1"], lambd)
        params = update_params(params, grads)
        if (i % 100 == 0):
            print("cost after " + str(i) + ": ", cost)

    return params

def predict(X, params):
    a = forward_step(params, X)
    return (a > 0.5) * 1

def get_correctness(predict, Y):
    total = Y.shape[1]
    wrong = np.sum(np.logical_xor(predict, Y))
    return (1 - wrong / total) * 100

params = model(X_train, Y_train, iterations=10000, lambd = 10)

train_correctness = get_correctness(predict(X_train, params), Y_train)
test_correctness = get_correctness(predict(X_test, params), Y_test)

print("logistic regression with L2 regularization")
print("train correctness: ", train_correctness, "%")
print("test correctness: ", test_correctness, "%")

#Cats vs noncats after 10K iterations, lambda = 1
#train correctness:  99.04306220095694 %
#test correctness:  72.0 %

#print(reshape_X(X_train_orig).shape)
#print(convert_y_labels(Y_train_orig))
#plt.imshow(X_train_orig[1])
#plt.show()

