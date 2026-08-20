#using my knowledge of gradient descent and vectorization to speed up the linear regression model 


import time
import numpy as np
import pandas as pd
from pathlib import Path

#just putting learning rate here for now, update for automatic later
alpha = 0.1

class MultipleLinearRegression:
    data: pd.DataFrame
    #takes a pandas dataframe as input and returns linear regression model weights 
   
    def __init__(self, data):
        self.data = data


    def run(self):
        datapoints = self.data

        #mX means matrix X, converts to data matrix
        mX = datapoints.to_numpy()

        #changes to float for easier operations
        mX = np.astype(mX, float)

        mX = self.__normalizer(mX)

       
        
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
        
                self.inverse_norm[1,1] = 1
                return (self.y_scale)*(beta @ self.inverse_norm)

        


    def __get_derivative(self, matrix:np.ndarray, beta:np.ndarray, targets:np.ndarray):

       
       
        #calculates (w*x+b for each training example, subtracts the targets)
        error = (beta @ matrix) - targets
        
        #returns a np vector with the partial derivatives of the cost function for w1...wn and b (used the chain rule)
        derivative = (1/matrix.shape[1])*(error @ matrix.transpose())
        return derivative

    def __normalizer(self, data):

        #using max normalization, can just multiply by its inverse instead of making a new function(scaled target)

        norm_ar = np.zeros(data.shape[1])
        for i in range(data.shape[1]):
            norm_ar[i] = (max(data[:,i]))

        self.y_scale = float(norm_ar[-1])
        

        
        diag = np.diag(1/norm_ar)
        self.inverse_norm = diag
        
        return data @ (diag)


        

        
a = time.time()    
df = pd.read_csv("week2/data.csv")
x = MultipleLinearRegression(df)
print(x.run())
b = time.time()
print(b-a)