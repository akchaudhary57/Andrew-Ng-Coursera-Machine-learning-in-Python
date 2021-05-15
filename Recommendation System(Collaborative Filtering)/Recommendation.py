import numpy as np

class Recommendation_System:
    def __init__(self):
        pass

    def transform_X_Theta(self,params,num_users,num_movies,num_features):
        X = params[:num_movies*num_features].reshape(num_movies,num_features)
        Theta = params[num_movies*num_features:].reshape(num_users,num_features)
        return X,Theta

    def CostFunction(self,params,X,Y,Theta,R,num_users,num_movies,num_features):
        X,Theta = self.transform_X_Theta(params,num_users,num_movies,num_features)
        pred = X@Theta.T
        error = (pred-Y)
        J = 1/2 * np.sum((error**2)*R)
        return J

    def Reg_CostFunction(self,params,X,Y,Lambda_C,Theta,R,num_users,num_movies,num_features):
        X,Theta = self.transform_X_Theta(params,num_users,num_movies,num_features)
        J = self.CostFunction(params,X,Y,Theta,R,num_users,num_movies,num_features)
        reg_x = Lambda_C/2* np.sum(Theta**2)
        reg_theta = Lambda_C/2* np.sum(X**2)
        reg_j= J + reg_x+ reg_theta
        return reg_j

    def Compute_Gradient(self,params,X,Theta,R,Y,num_users,num_movies,num_features):
        X,Theta = self.transform_X_Theta(params,num_users,num_movies,num_features)
        pred = X@Theta.T
        error = (pred-Y)
        X_grad = error*R@Theta
        X_theta = (error*R).T@X
        grad = np.append(X_grad.flatten(),X_theta.flatten())
        return X_grad,X_theta,grad

    def Regularized_Gradient(self,params,X,Y,Lambda_C,Theta,R,num_users,num_movies,num_features):
        X,Theta = self.transform_X_Theta(params,num_users,num_movies,num_features)
        X_grad,X_theta,grad = self.Compute_Gradient(params,X,Theta,R,Y,num_users,num_movies,num_features)
        reg_X_grad = X_grad+Lambda_C*X
        reg_X_theta = X_theta+Lambda_C*Theta
        reg_grad = np.append(reg_X_grad.flatten(),reg_X_theta.flatten())
        return reg_grad
