import pandas as pd
import numpy as np
from numpy.linalg import svd


class pca:
    def __init__(self):
        pass

    def featurenormalization(self,X):
         mean = np.mean(X,axis = 0)
         std = np.std(X,axis=0)
         Z = (X-mean)/std
         return Z,mean,std

    def pca(self,X):
        m,n = X.shape
        sigma = 1/m*(X.T@X)
        U,S,V = svd(sigma)
        return U,S,V,sigma


    def projectdata(self,X,U,k):
        m = X.shape[0]
        U_reduced = U[:,:k]
        Z = np.zeros((m,k))
        for i in range(m):
            for j in range(k):
                Z[i,j] = X[i,:] @ U_reduced[:,j]
        return Z

    def recoverdata(self,Z,U,k):
        m,n = Z.shape[0],U.shape[0]
        U_reduced = U[:,:k]
        x_rec = np.zeros((m,n))
        for i in range(m):
                x_rec[i,:] = Z[i,:] @ U_reduced.T
        return x_rec
