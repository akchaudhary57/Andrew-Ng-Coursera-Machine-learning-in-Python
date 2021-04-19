import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self,K):
        self.K = K

    def findClosestCentroids(self,X,centriods):
        """ this function will return the idx of closest distance from centriods """
        m,n = X.shape
        k = len(centriods)
        idx =  np.zeros((m,1))
        for i in range(m):
            displaydata = np.zeros((1,self.K))
            for j in range(k):
                 displaydata[:,j] = np.sqrt(np.sum(np.power((X[i,:]-centriods[j,:]),2)))
            idx[i] = np.argmin(displaydata)+1
        return idx


    def computeCentroids(self,X,idx):
        m,n = X.shape
        centroids = np.zeros((self.K,n))
        count = np.zeros((self.K,1))
        for i in range(m):
            index = int(idx[i]-1)
            centroids[index,:] += X[i,:]
            count[index]+=1
        return centroids/count

    def plot_KMeans(self,X,idx,centroid,num_iter):
        """ this function will visualize Kmeans clustering """
        m,n =X.shape
        fig,ax = plt.subplots(nrows = num_iter,ncols =1,figsize = (6,36))

        for i in range(num_iter):
            color = 'rbg'
            for k in range(self.K+1):
                grp = (idx == k).reshape(m,1)
                ax[i].scatter(X[grp[:,0],0],X[grp[:,0],1], c = color[k-1],s=15)
            ax[i].scatter(centroid[:,0],centroid[:,1],c = "Black",s = 150,linewidth = 3,marker = 'x')
            centroid = self.computeCentroids(X,idx)
            idx = self.findClosestCentroids(X,centroid)
            title = 'No of iteration'+' '+str(i)
            ax[i].set_title(title)


    def kMeansInitCentroids(self,X):
        m,n = X.shape
        centriod = np.zeros((self.K,n))
        for i in range(self.K):
            centriod[i] = X[np.random.randint(0,m+1),:]
        return centriod

    def runKmeans(self,X,centroid,num_iter):
        idx = self.findClosestCentroids(X,centroid)
        for i in range(num_iter):
            centroid =  self.computeCentroids(X,idx)
            idx = self.findClosestCentroids(X,centroid)
        return centroid,idx
