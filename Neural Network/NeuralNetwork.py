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
from  Logistic_Regression_Scratch import *

class NeuralNetwork(LogisticRegression):
    def __init__(self):
        pass

    def FeedForward(self,X,theta1,theta2):
        a1 = OneVsall.append_one(self,X)
        z2 = a1@theta1.T
        a2 = LogisticRegression.sigmoid(self,z2)
        a2 = np.append(np.ones((a2.shape[0],1)),a2,axis = 1)
        z3 = a2@theta2.T
        a3 = LogisticRegression.sigmoid(self,z3)
        return a3,a2,a1,z2


    def transform_theta(self,nn_params,input_layer_size,hidden_layer_size,num_labels):
        Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

        Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))
        return Theta1,Theta2

    def transform_y(self,y,num_labels):
        y_matrix = y.reshape(-1)
        y_matrix = np.eye(num_labels)[y_matrix]
        return y_matrix

    def nncostfunction(self,nn_params,input_layer_size,hidden_layer_size,num_labels,X, y, lambda_= 0.0):
            Theta1,Theta2 = self.transform_theta(nn_params,input_layer_size,hidden_layer_size,num_labels)
            m = y.shape[0]
            n = X.shape[1]
            J = 0
            Theta1_grad = np.zeros(Theta1.shape)
            Theta2_grad = np.zeros(Theta2.shape)
            a3,a2,a1,z2 = self.FeedForward(X,Theta1,Theta2)
            y_matrix = self.transform_y(y,num_labels)
            J = (1/m) * sum(sum((-y_matrix)* np.log(a3) - (1-y_matrix) * np.log(1-a3)));
            Regularization = (lambda_/(2*m)* (sum(sum(Theta1[:,1:]*Theta1[:,1:])) + sum(sum(Theta2[:,1:]*Theta2[:,1:]))))
            J = J + Regularization
            grad = self.BackPropagation(nn_params,input_layer_size,hidden_layer_size,num_labels,X, y, lambda_)
            return J,grad

    def randInitializeWeights(self,L_in,L_out,epsilon_init = 0.12):
        W = np.zeros((L_out,1+L_in))
        W = np.random.rand(L_out,1+L_in)*2*epsilon_init - epsilon_init
        return W


    def sigmoidGradient(self,z):
        z1 = LogisticRegression.sigmoid(self,z)
        z2 = 1- LogisticRegression.sigmoid(self,z)
        g= z1*z2
        return g

    def BackPropagation(self,nn_params,input_layer_size,hidden_layer_size,num_labels,X, y, lambda_= 0.0):
        m,n = X.shape
        theta1,theta2 = self.transform_theta(nn_params,input_layer_size,hidden_layer_size,num_labels)
        Theta1_grad = np.zeros(theta1.shape)
        Theta2_grad = np.zeros(theta2.shape)
        a3,a2,a1,z2 =  self.FeedForward(X,theta1,theta2)
        y_new = self.transform_y(y,num_labels)
        # Step 1, calculate delta3
        delta3 = a3 - y_new
        # Step 2, calcualte delta2 with sigmoid gradient
        delta2 = (delta3@theta2[:,1:]) * self.sigmoidGradient(z2)
        # Step 3, calculate Theta2_grad and Theta3_grad
        Theta2_grad = Theta2_grad+delta3.T@a2
        Theta1_grad = Theta1_grad+delta2.T@a1
        # Step 4, divide by m
        Theta1_grad = 1/m*Theta1_grad
        # Step 5, add regularization
        Theta1_grad[:,1:] = Theta1_grad[:,1:] + (lambda_/m)* theta1[:,1:]
        Theta2_grad = 1/m*Theta2_grad
        Theta2_grad[:,1:] = Theta2_grad[:,1:] + (lambda_/m)* theta2[:,1:]
        grad = np.concatenate([Theta1_grad.ravel(),Theta2_grad.ravel()])
        return grad



    def debugInitializeWeights(self,fan_out, fan_in):
        """
        Initialize the weights of a layer with fan_in incoming connections and fan_out outgoings
        connections using a fixed strategy. This will help you later in debugging.
        Note that W should be set a matrix of size (1+fan_in, fan_out) as the first row of W handles
        the "bias" terms.
        Parameters
        ----------
        fan_out : int
            The number of outgoing connections.
        fan_in : int
            The number of incoming connections.
        Returns
        -------
        W : array_like (1+fan_in, fan_out)
            The initialized weights array given the dimensions.
        """
        # Initialize W using "sin". This ensures that W is always of the same values and will be
        # useful for debugging
        W = np.sin(np.arange(1, 1 + (1+fan_in)*fan_out))/10.0
        W = W.reshape(fan_out, 1+fan_in, order='F')
        return W


    def computeNumericalGradient(self,J, theta, e=1e-4):
        """
        Computes the gradient using "finite differences" and gives us a numerical estimate of the
        gradient.
        Parameters
        ----------
        J : func
            The cost function which will be used to estimate its numerical gradient.
        theta : array_like
            The one dimensional unrolled network parameters. The numerical gradient is computed at
             those given parameters.
        e : float (optional)
            The value to use for epsilon for computing the finite difference.
        Notes
        -----
        The following code implements numerical gradient checking, and
        returns the numerical gradient. It sets `numgrad[i]` to (a numerical
        approximation of) the partial derivative of J with respect to the
        i-th input argument, evaluated at theta. (i.e., `numgrad[i]` should
        be the (approximately) the partial derivative of J with respect
        to theta[i].)
        """
        numgrad = np.zeros(theta.shape)
        perturb = np.diag(e * np.ones(theta.shape))
        for i in range(theta.size):
            loss1, _ = J(theta - perturb[:, i])
            loss2, _ = J(theta + perturb[:, i])
            numgrad[i] = (loss2 - loss1)/(2*e)
        return numgrad

    def checkNNGradients(self,nncostfunction, lambda_=0):
        """
        Creates a small neural network to check the backpropagation gradients. It will output the
        analytical gradients produced by your backprop code and the numerical gradients
        (computed using computeNumericalGradient). These two gradient computations should result in
        very similar values.
        Parameters
        ----------
        nnCostFunction : func
            A reference to the cost function implemented by the student.
        lambda_ : float (optional)
            The regularization parameter value.
        """
        input_layer_size = 3
        hidden_layer_size = 5
        num_labels = 3
        m = 5

        # We generate some 'random' test data
        Theta1 = self.debugInitializeWeights(hidden_layer_size, input_layer_size)
        Theta2 = self.debugInitializeWeights(num_labels, hidden_layer_size)

        # Reusing debugInitializeWeights to generate X
        X = self.debugInitializeWeights(m, input_layer_size - 1)
        y = np.arange(1, 1+m) % num_labels
        # print(y)
        # Unroll parameters
        nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])

        # short hand for cost function
        costFunc = lambda p: self.nncostfunction(p, input_layer_size, hidden_layer_size,
                                            num_labels, X, y, lambda_)
        cost, grad = costFunc(nn_params)
        numgrad = self.computeNumericalGradient(costFunc, nn_params)
        print('numgrad',numgrad.shape)
        print('grad',grad.shape)
        # Visually examine the two gradient computations.The two columns you get should be very similar.
        print(np.stack([numgrad, grad], axis=1))
        print('The above two columns you get should be very similar.')
        print('(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

        # Evaluate the norm of the difference between two the solutions. If you have a correct
        # implementation, and assuming you used e = 0.0001 in computeNumericalGradient, then diff
        # should be less than 1e-9.
        diff = np.linalg.norm(numgrad - grad)/np.linalg.norm(numgrad + grad)

        print('If your backpropagation implementation is correct, then \n'
              'the relative difference will be small (less than 1e-9). \n'
              'Relative Difference: %g' % diff)


    def predict(self,Theta1, Theta2, X):
        """
        Predict the label of an input given a trained neural network
        Outputs the predicted label of X given the trained weights of a neural
        network(Theta1, Theta2)
        """
        # Useful values
        m = X.shape[0]
        num_labels = Theta2.shape[0]

        # You need to return the following variables correctly
        p = np.zeros(m)
        h1 = LogisticRegression.sigmoid(self,np.dot(np.concatenate([np.ones((m, 1)), X], axis=1), Theta1.T))
        h2 = LogisticRegression.sigmoid(self,np.dot(np.concatenate([np.ones((m, 1)), h1], axis=1), Theta2.T))
        p = np.argmax(h2, axis=1)
        return p
