import time
import numpy as np
import pandas as pd
from pathlib import Path

class MultipleLinearRegression:
    data: pd.DataFrame
    #takes a pandas dataframe as input and returns linear regression model weights 
   
    def __init__(self, data):
        self.data = data


    def run(self):
        datapoints = self.data
        #mX means matrix X, converts to data matrix
        mX = datapoints.to_numpy()
        print(type(mX))
        


    


df = pd.read_csv("week2/data.csv")
x = MultipleLinearRegression(df)
x.run()