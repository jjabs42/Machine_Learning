#using my knowledge of gradient descent and vectorization to speed up the linear regression model 


import time
import numpy as np
import pandas as pd
from pathlib import Path

#just putting learning rate here for now, update for automatic later
alpha = 0.03

class MultipleLinearRegression:
    data: pd.DataFrame
    #takes a pandas dataframe as input and returns linear regression model weights 
   
    def __init__(self, data):
        self.data = data


    def run(self):
        datapoints = self.data

        #mX means matrix X, converts to data matrix
        mX = datapoints.to_numpy()
        
        targets = mX[:,1]

        #assigns last vector to 1's, to help calculate the derivative of the bias and multiply the bias by 1
        oneMatrix = mX.copy().transpose()
        oneMatrix[-1, :] = 1
        
        #creates an array of the starting weights, and the value of the bias
        beta = np.zeros(mX.shape[1])
        

        for _ in range(10000):

            old_beta = beta.copy()
            
           #gets derivative
            derivative = self.__get_derivative(oneMatrix, beta, targets)
            
    
            #scales derivative vector by learning rate
            derivative*=alpha
            
            #updates weights
            beta-=derivative

            if np.abs(np.max(old_beta-beta)) < 1e-5:
                return beta

        


    def __get_derivative(self, matrix:np.ndarray, beta:np.ndarray, targets:np.ndarray):

       
       
        #calculates (w*x+b for each training example, subtracts the targets)
        error = (beta @ matrix) - targets
        
        #returns a np vector with the partial derivatives of the cost function for w1...wn and b (used the chain rule)
        derivative = (1/matrix.shape[1])*(error @ matrix.transpose())
        return derivative

        

        
    
df = pd.read_csv("week2/data.csv")
x = MultipleLinearRegression(df)
print(x.run())
