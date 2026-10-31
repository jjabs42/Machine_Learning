#using my knowledge of gradient descent and vectorization to speed up the linear regression model 


import time
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

#just putting learning rate here for now, update later with better approach
alpha = 0.05


class MultipleLinearRegression:
    data: pd.DataFrame
    #takes a pandas dataframe as input and returns linear regression model weights 
   
    def __init__(self, data):
        self.data = data
        
        


    def run(self):
        datapoints = self.data
    
        
        #mX means matrix X, converts to data matrix
        mX = datapoints.to_numpy(copy=True)

        #changes to float for easier operations
        mX = np.astype(mX, float)

        mX = self._normalizer(mX)

       
        
        targets = mX[:,-1]

        #assigns last vector to 1's, to help calculate the derivative of the bias and multiply the bias by 1
        oneMatrix = mX.copy().transpose()
        oneMatrix[-1, :] = 1
        
        #creates an array of the starting weights, and the value of the bias
        beta = np.zeros(mX.shape[1])
        
        #place where weights update
        for i in range(10000):

            old_beta = beta.copy()
            
           #gets derivative
            derivative = self._get_derivative(oneMatrix, beta, targets)
            
    
            #scales derivative vector by learning rate
            derivative*=alpha
            
            #updates weights
            beta-=derivative

            print(self._cost(oneMatrix,beta,targets))
            

            if np.max(np.abs(old_beta-beta)) < 1e-6:
        
                self.inverse_norm[-1,-1] = 1
                print(f"i = {i}")
                return (self.y_scale)*(beta @ self.inverse_norm)
        

        


    def _get_derivative(self, matrix:np.ndarray, beta:np.ndarray, targets:np.ndarray):

       
       
        #calculates (w*x+b for each training example, subtracts the targets)
        error = (beta @ matrix) - targets
        
        #returns a np vector with the partial derivatives of the cost function for w1...wn and b (used the chain rule)
        derivative = (1/matrix.shape[1])*(error @ matrix.transpose())
        return derivative

    def _normalizer(self, data):

        #using max normalization, can just multiply by its inverse instead of making a new function(scaled target)
        #there may be some cases where it doesn't work I still have to fix this for a more robust solution

        norm_ar = np.zeros(data.shape[1])
        for i in range(data.shape[1]):
            norm_ar[i] = (max(data[:,i]))

        self.y_scale = float(norm_ar[-1])
        

        
        diag = np.diag(1/norm_ar)
        
        self.inverse_norm = diag
        
        return data @ (diag)

    #getting cost function so can plot the iterations
    
    def _cost(self, matrix: np.ndarray, beta: np.ndarray, targets: np.ndarray):
        # beta @ matrix calculates (w * x + b) for all rows instantly using the bias trick
        y_hat = beta @ matrix
        
        # Calculate the normalized error 
        error = y_hat - targets

       
        
        # squared summation
        return (1/(2*matrix.shape[1])) * (np.dot(error, error))



        
if __name__ == "__main__":
    a = time.time()    
    df = pd.read_csv("week2/physicsLab.csv")
    x = MultipleLinearRegression(df)
    print(x.run())
    b = time.time()
    print(b-a)