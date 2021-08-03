import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#tf.compat.v1.logging.set_verbosity(10)
tf.get_logger().setLevel('INFO')

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train = pd.read_csv('datasets/iris/iris_training.csv', names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv('datasets/iris/iris_test.csv', names=CSV_COLUMN_NAMES, header=0)
train_y = train.pop('Species')
test_y = test.pop('Species')


def input_fn(features, labels, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)

my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

classifier = tf.estimator.DNNClassifier(hidden_units=[30,30], feature_columns=my_feature_columns, n_classes=len(SPECIES))

classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000
)