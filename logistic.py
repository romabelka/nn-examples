import numpy as np
import improv_utils
import matplotlib.pyplot as plt

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = improv_utils.load_dataset()

#plt.imshow(X_train_orig[1])
#plt.show()

print(X_train_orig.shape)
