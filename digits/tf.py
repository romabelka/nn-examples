#https://www.kaggle.com/c/digit-recognizer
import numpy as np
import n_utils
import matplotlib.pyplot as plt
import tensorflow.keras as ks

n_labels = 10
MAX_EXP_ARG = 500
keep_prob = 0.8

data = np.genfromtxt('digits/datasets/train.csv', delimiter=',')

X = data.T[1:, 1:] / 255.
Y = data.T[0:1, 1:].astype(int)

m = X.shape[1]
m_test = m // 5
m_train = m - m_test

X_train = X[:, :m_train]
Y_train = Y[:, :m_train]

X_test = X[:, m_train :]
Y_test = Y[:, m_train :]

n_x = X.shape[0]

print("m_train", m_train, ". m_test", m_test)

model = ks.models.Sequential([
    ks.layers.Dense(28*28),
    ks.layers.Dense(60, activation='relu'),
    ks.layers.Dropout(0.2),
    ks.layers.Dense(40, activation='relu'),
    ks.layers.Dropout(0.2),
    ks.layers.Dense(30, activation='relu'),
    ks.layers.Dropout(0.2),
    ks.layers.Dense(30, activation='relu'),
    ks.layers.Dropout(0.2),
    ks.layers.Dense(10)
])

model.compile(
    optimizer='adam', 
    loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True), 
    metrics=['accuracy']
)

model.fit(x=X_train.T, y=Y_train.T, epochs=100)
model.evaluate(x=X_test.T, y=Y_test.T, batch_size=None)

h_layers = [60, 40, 30, 30]

#tune_hyper_params(architectures=[h_layers], lambds=[0], keep_probs = [.8], iterations=300, learning_rate=0.004)
def imshow(X, i):
    plt.imshow(X[:, i].reshape(28,28))
    plt.show()

def save_test_data():
    test_data = np.genfromtxt('digits/datasets/test.csv', delimiter=',')
    #test_data = np.genfromtxt('digits/datasets/test_debug.csv', delimiter=',')
    X_result = test_data[1:, :]

    res = np.argmax(model.predict(x=X_result), axis=1)

    submit_data = np.concatenate((
        np.arange(1, res.shape[0] + 1).reshape(-1, 1), 
        res.reshape(-1, 1)
    ), axis=1)

    np.savetxt('digits/datasets/submition.csv', submit_data.astype(int), fmt='%i', delimiter=',', header='ImageId,Label')

save_test_data()
#save_test_data(params, h_layers)

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