import tensorflow as tf
import tensorflow.keras.layers as layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import experimental
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.pooling import MaxPool2D

train_data = pd.read_csv('./digits/datasets/train.csv', header=0)

labels = train_data.pop('label').to_numpy().reshape(-1, 1)
input = train_data.to_numpy().reshape(-1, 28, 28, 1)

validation_len = int(input.shape[0] * 0.1)

test_input = input[:validation_len, :, :, :]
test_labels = labels[:validation_len, :]

train_input = input[validation_len:, :, :, :]
train_labels = labels[validation_len:, :]


print(train_input.shape, train_labels.shape, test_input.shape, test_labels.shape)

#model = tf.keras.Sequential([
#    layers.experimental.preprocessing.Rescaling(1./255),
#    layers.Conv2D(filters=16, kernel_size=3, activation='relu'), # 16 x 26
#    layers.BatchNormalization(),
#    layers.Dropout(rate=0.2),
#    layers.Conv2D(filters=32, kernel_size=5, activation='relu'), # 32 x 22
#    layers.MaxPool2D(), # 32 X 11
#    layers.BatchNormalization(),
#    layers.Dropout(rate=0.2),
#    layers.Conv2D(filters=64, kernel_size=3, activation='relu'), # 64 x 9 
#    layers.MaxPool2D(), # 64 x 5 (4?)
#    layers.BatchNormalization(),
#    layers.Dropout(rate=0.2),
#    layers.Flatten(),
#    layers.Dense(128, activation='relu'),
#    layers.Dense(10)
#])

model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),

    layers.Conv2D(filters=64, kernel_size=5, padding='same', activation='relu'),
    layers.BatchNormalization(),

    layers.Conv2D(filters=64, kernel_size=5, padding='same', activation='relu'),
    layers.BatchNormalization(),

    layers.MaxPool2D(),
    layers.Dropout(rate=0.25),

    layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),

    layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    layers.Dropout(rate=0.25),

    layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.25),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.25),

    layers.Dense(10)
])


model.compile(
    optimizer='adam',
    loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

#model.build(train_input.shape)

model.fit(x= train_input, y = train_labels, batch_size=32, epochs=15)
model.evaluate(x= test_input, y = test_labels)

submit_data = pd.read_csv('./digits/datasets/test.csv', header=0)
submit_input = submit_data.to_numpy().reshape(-1, 28, 28, 1)

#plt.imshow(submit_input[0, :, :, :])
#plt.show()

cls = np.argmax(model.predict(x=submit_input), axis=-1)

submit_data = np.concatenate((np.arange(1, cls.shape[0] + 1).reshape(-1,1), cls.reshape(-1,1)), axis=1)
np.savetxt('digits/datasets/submition.csv', submit_data.astype(int), fmt='%i', delimiter=',', header='ImageId,Label')

#model.summary()
