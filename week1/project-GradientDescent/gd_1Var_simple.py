import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

#configs:
learning_rate = ...

df = pd.read_csv("data.csv")

x_array = pd.Series.to_numpy(df["x"])
y_array = pd.Series.to_numpy(df["y"])
#returns the partial derivitives d/dw and d/db for the cost function, at points w and b
def derivitive(x, y, w, b) -> tuple:
    

    shape = x.shape[0]

    #finds w and b
    der_w = 0
    der_b = 0
    for i in range(shape):
        der_w+=(w*x[i]+b-y[i])*x[i]
        der_b+=(w*x[i]+b-y[i])
    der_w/=shape
    der_b/=shape

    return(float(der_w), float(der_b))

w = -1
b = 9
print(derivitive(x_array, y_array, w, b))

plt.plot(x_array, y_array, "ro")
plt.axline((0,b), slope=w)
plt.show()










