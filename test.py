import numpy as np


def softmax(X):       # [256, 10]
    X_item = [float(np.exp(item)) for item in X]
    X_sum = float(np.sum(X_item))
    X_item = [item/X_sum for item in X_item]
    return X_item

print(softmax([100, 101, 102]), '\n')

print('a', softmax([10.0, 10.1, 10.2]))
print('b', softmax([-100, -101, -102]))
print('c', softmax([-2, -1, 0]))
print('d', softmax([1000, 1010, 1020]))


