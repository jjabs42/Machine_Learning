import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])


def cost(x_data, y_data, w, b):
    sums = 0
    for i in range(len(x_data)):
        sums += ((w*x_data[i] + b) - y_data[i])**2

    sample_size = len(x_data)
    return sums/2*sample_size

print(cost(x_train, y_train, 200 , 100))

                