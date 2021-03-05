import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import fmin
from scipy.optimize import minimize
import scipy.optimize as opt
import sys
sys.path.insert(1, 'C:\\Users\\AjitC\\Desktop\\coursera\\machine-learning-ex1\\ex2\\ex2')
from scipy.optimize import minimize, fmin_tnc
from  Logistic_Regression_Regularization import *
from  Logistic_Regression_Scratch import *


class OneVsall(logistic_regression_reg,LogisticRegression):
    def __init__(self):
        pass

    def displaydata(self,X,width = None):
        m,n = X.shape
        width = int(np.round(np.sqrt(n)))
        height = n/width
        display_row = int(np.floor(np.sqrt(m)))
        display_cols = int(np.ceil(m/display_row))
        fig,arra = plt.subplots(nrows = display_row,ncols = display_row, figsize = (10,10))
        for i,ax in enumerate(arra.flat):
            ax.imshow(X[i].reshape(width,width,order = 'F'),cmap = 'Greys')
            ax.axis('off')

    def append_one(self,X):
        m,n = X.shape
        X0 = np.ones((m,1))
        X = np.append(X0,X,axis = 1)
        return X

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

    def minimumcost(self,X,y,num_labels,lambda_coff):
        m,n = X.shape
        all_theta = np.zeros((num_labels,n))
        for i in range(num_labels):
            digits = i if i else 10
            all_theta[i] = opt.fmin_cg(f = self.costfunction, x0 = all_theta[i],fprime = self.gradient,args = (X,(y == digits).flatten(),lambda_coff),maxiter = 50)
        return all_theta

    def Onehot_Encoding(self,y,num_labels):
        yb = np.zeros((y.shape[0],num_labels))
        for i in range(y.shape[0]):
            for j in range(1,num_labels+1):
                if y[i] == j:
                    yb[i,j-1] = 1
        return yb
