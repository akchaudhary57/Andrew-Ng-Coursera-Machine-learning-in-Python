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
