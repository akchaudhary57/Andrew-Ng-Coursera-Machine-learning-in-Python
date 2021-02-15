import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin

class LogisticRegression:
    def __init__(self):
        pass
    def sigmoid(self,z):
        zt = 1.0/(1+np.exp(-z))
        return zt
    def hypothesis(self,X,theta):
        z = np.dot(X,theta)
        htheta = self.sigmoid(z)
        return htheta

    def costfunction(self,theta,X,y):
        m,n = X.shape
        htheta = self.hypothesis(X,theta)
        J = (-1/m) * np.sum(np.multiply(y, np.log(htheta)) + np.multiply((1-y), np.log(1 - htheta)))
        return J

    def gradient(self,X,y,theta):
        m,n = X.shape
        htheta =  self.hypothesis(X,theta)
        grad = ((htheta -y)@X)/m
        return grad

    def minimumcost(self,costfunction,theta,X,y):
        min_cost = fmin(func = self.costfunction,x0 = theta, args = (X,y))
        return min_cost

    def func(self,x):
        x = np.where(x<0.5,0,1)
        return x

    def predict(self,X,theta):
        ypred = self.hypothesis(X,theta)
        ypred = self.func(ypred)
        return ypred

    def score(self,y,ypred):
        Acc = np.mean(y==ypred)
        return Acc
