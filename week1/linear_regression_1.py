import matplotlib.pyplot as plt
import numpy as np

x = np.array([1.0, 2.0])
y = np.array([300.0, 500.0])

index = 1

x_i = x[index]
y_i = y[index]

def train(x, w, b):
    z = x.shape[0]
    m = np.zeros(z)


    for i in range(z):
        print(1)
        m[i] = w*x[i] + b

    return m
dfsa =  train(x, 100, 100)

plt.plot(x, dfsa)
plt.scatter(x, y, marker="x", c="r")
plt.show()

