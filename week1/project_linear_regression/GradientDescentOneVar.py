#simple linear regression using gradient descent - not using numpy calculations
import time
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

#configs:
learning_rate = 0.001

df = pd.read_csv("week1/project_linear_regression/data.csv")

x_array = pd.Series.to_numpy(df["x"])
y_array = pd.Series.to_numpy(df["y"])
#returns the partial derivatives d/dw and d/db for the cost function, at points w and b
def derivative(x, y, w, b) -> tuple:
    
    shape = x.shape[0]

    #finds w and b
    der_w = 0
    der_b = 0
    for i in range(shape):
        der_w+=(w*float(x[i])+b-float(y[i]))*float(x[i])
        der_b+=(w*float(x[i])+b-float(y[i]))
    der_w/=shape
    der_b/=shape

    return(float(der_w), float(der_b))


#takes in the parameters and returns the tuple of the ideal w, b
def gradient_descent(x, y, w, b, learning_rate) -> tuple:
    while True:
        dw, db = derivative(x, y, w, b)
        w -=learning_rate*dw
        b -=learning_rate*db
        if abs(dw) < 1e-6 and abs(db) < 1e-6:
            return w, b
            
            
    
def cost(x, y, w, b):
    shape = x.shape[0]
    sum = 0
    for i in range(shape):
        sum += (w*x[i] + b - y[i])**2
    return sum / (2*shape)


def main():
    w, b = gradient_descent(x_array, y_array, 0, 0, learning_rate)
    print(w, b)



    plt.plot(x_array, y_array, "ro")
    plt.axline((0,b), slope=w)
    plt.show()
    return (w, b)


if __name__ == "__main__":
    main()







