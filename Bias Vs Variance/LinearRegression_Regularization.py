import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import fmin
from scipy.optimize import minimize

class LinearRegression:
    def __init__(self):
        pass
    def featureNormalization(self,X):
        mu = np.mean(X,axis = 0)
        X_norm = X - mu
        sigma = np.std(X_norm, axis=0)
        X_norm /= sigma
        return  X_norm, mu, sigma

    def hypothesis(self,X,theta):
        htheta = np.dot(X,theta.T)
        return htheta

    def gradientDescent(self,X,y,theta,lambda_):
            m = len(X)
            htheta = self.hypothesis(X,theta)
            J = np.sum(((htheta-y.T)**2))/(2*m)
            Regularization = lambda_/(2*m) * (np.sum(theta[1:]*theta[1:]))
            J = J+Regularization
            grad = (np.dot((htheta-y.T),X))/m
            grad[:,1:] =  grad[:,1:] + (lambda_/(m)) * theta[1:]
            return J,grad

    def minimumcost(self,gradientDescent,X,y,theta,lambda_, maxiter=200):
        initial_theta = np.zeros(X.shape[1])
        costFunction = lambda t: self.gradientDescent(X, y, t, lambda_)
        options = {'maxiter': maxiter}
        res = optimize.minimize(costFunction, initial_theta, jac=True, method='TNC', options=options)
        return res.x
