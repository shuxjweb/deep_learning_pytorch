import numpy as np


a = [2.33, 1.07, 1.23]
b = [3.14, 0.98, 1.32]


def squared_loss(y_hat, y):     # [10, 1], [10,]
    return (y_hat - y) ** 2 / 2

z = []
for x, y in zip(a, b):
    z.append(squared_loss(x, y))
print(np.mean(z))


