import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import fmin
from scipy.optimize import minimize
import scipy.optimize as opt
import sys
from OneVsAllLogisticRegression import *
from  Logistic_Regression_Regularization import *
from  logistic_regression import *

class FeedForward_NN(LogisticRegression):
    def __init__(self):
        pass

    def FeedForward(self,X,theta1,theta2):
        a1 = X@theta1.T
        z1 = LogisticRegression.sigmoid(self,a1)
        h0 = np.ones((z1.shape[0],1))
        z1 = np.append(h0,z1,axis = 1)
        a2 = z1@theta2.T
        z2 = LogisticRegression.sigmoid(self,a2)
        return z2
