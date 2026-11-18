import numpy as np
from pandas import DataFrame as df


class LogisticRegression:
    def __init__(self, data: df):
        # converts pandas dataframe to a matrix
        dataMatrix = data.to_numpy()
        self.dataMatrix = dataMatrix


    def runLogisticRegression(self):
        dataMatrix = self.dataMatrix


    # gets derivative of cost function
    def getDerivative(self) -> np.ndarray:
        ...

    # reminder to write down matrix manipulations
    def sigmoidFunction(self, weights: np.ndarray, bias, xValues: np.ndarray) -> int:
        innerFunction = weights * xValues + bias




if __name__ == "__main__":
    ...