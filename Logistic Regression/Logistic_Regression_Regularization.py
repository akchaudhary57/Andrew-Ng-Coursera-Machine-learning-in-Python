import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from scipy.optimize import minimize
from Logistic_Regression_Scratch import *

class logistic_regression_reg(LogisticRegression):
    def __init__(self):
        pass
    def mapfeature(self,X,degree):
        data1 = X.copy()
        data1.insert(0,'Ones',1)
        for i in range(1,degree+1):
            for j in range(0,i+1):
                data1['x'+str(i)+str(j)] = np.power(data1[0],i-j)*np.power(data1[1],j)
        cols = [1,2]
        data1.drop(data1.columns[cols],axis = 1,inplace = True)
        return data1

    def costfunction(self,theta,X,y,lambda_coff):
        m,n = X.shape
        J = LogisticRegression.costfunction(self,theta,X,y)
        J = J+ (lambda_coff/(2*m) * sum(theta[1:n] * theta[1:n]))
        return J

    def gradient(self,theta,X,y,lambda_coff):
        m,n = X.shape
        theta1 = theta[1:n]
        thetaF = np.hstack([0,theta1])
        grad = LogisticRegression.gradient(self,X,y,theta)
        gradF = grad + lambda_coff/m*thetaF
        return gradF

    def minimumcost(self,costfunction,theta,X,y,lambda_coff):
       min_cost = minimize(fun = costfunction, x0 = theta, args = (X,y,lambda_coff))
       theta_opt = min_cost.x
       return theta_opt
