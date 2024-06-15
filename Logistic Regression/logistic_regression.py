''' Logistic Regression Model '''

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression():
    # Initializing some useful variables
    def __init__(self,lr=0.5,lamda=1e-4,num_iters=500):
        ''' '__init__' takes arguments as Learning Rate(lr), Regularization Constant(lamda), Number of Iterations(num_iters)'''
        # All these parameters have been initialized by their default values, they can be changed when required
        self.lr = lr
        self.lamda = lamda
        self.num_iters = num_iters

    ''' Feature scaling using Standardization Technique '''
    def feature_scale(self,X):
        for i in range(X.shape[1]):
            X[:,i] = (X[:,i] - np.mean(X[:,i])) / (np.std(X[:,i]) + 1e-5)
        return X

    ''' Sigmoid function maps all the values in the range of 0 to 1 '''
    def sigmoid(self,z):
        return 1 / (1 + np.exp(-1*z))

    ''' It takes X and w as its arguments and returns Hypothesis Value(h = Matrix Multiplication of X and w) '''
    def hypothesis_function(self,X,w):
        return (self.sigmoid(X @ w.T))

    ''' Cost method takes X,y and w as its arguments and returns cost value with regularization '''
    def cost_function(self,X,y,w):
        m,n = X.shape
        h = self.hypothesis_function(X,w)
        J = (-1 / m) * (np.sum(y * np.log(h + 1e-5) + (1 - y) * np.log(1 - h + 1e-5))) + (self.lamda / (2 * m)) * np.sum(w[:,1:] ** 2)
        return J

    ''' 'fit' function takes X,y and number of different classes as its arguments and applies Gradient Descent Algorithm to get weights '''
    def fit(self,X,y,n_class):
        X = self.feature_scale(X) # Feature Scaling
        X = np.hstack((np.ones((len(X),1)),X)) # Adds Bias column

        m,n = X.shape
        
        self.n_class = n_class

        y_class = np.zeros((m,self.n_class)) # Created a matrix of size m x n_class with all entries as zeros

        # Converts 'y' values to 'y_class' as matrix type 
        for i in range(m):
            y_class[i,y[i]] = 1
        
        # 'Jhistory' and 'iters' keeps track of cost with each iteration
        self.Jhistory,self.iters = [],[]

        # Values of weights have been initialized to zero
        self.w = np.zeros((self.n_class,n))

        # Applying Gradient Descent Algorithm
        for i in range(self.num_iters):
            w_temp = self.w
            w_temp[:,0:1] = 0

            h = self.hypothesis_function(X,self.w)
            self.w -= ((self.lr / m) * ((h - y_class).T @ X + self.lamda * w_temp))

            # Storing the value of Cost(J) in each iteration
            self.Jhistory.append(self.cost_function(X,y_class,self.w))
            self.iters.append(i)
    
    ''' Plotting the learning curve between cost and number of iterations '''
    def plot(self):
        plt.plot(self.iters,self.Jhistory,color='#00008B')
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost Function(J)")
        plt.title("Cost Function Vs Iterations")

    ''' It returns the predicted values given by the trained model '''
    def predict(self,X):
        X = self.feature_scale(X)
        X = np.hstack((np.ones((len(X),1)),X))
        ypred = self.hypothesis_function(X,self.w)
        ypred = np.argmax(ypred,axis=1).reshape(len(ypred),1)
        return ypred

    ''' Calculates and returns the probability of the sample for each class in the model '''
    def predict_prob(self,X):
        X = self.feature_scale(X)
        m,n = X.shape
        X = np.hstack((np.ones((m,1)),X))
        ypred_prob = self.hypothesis_function(X,self.w)
        return ypred_prob
    
    ''' It returns the accuracy of the trained model '''
    def accuracy(self,y,ypred):
        return (np.mean(y == ypred) * 100)

    ''' It returns the confusion matrix by taking y and ypred as its arguments '''
    def confusion_matrix(self,y,ypred):
        matrix = np.zeros((self.n_class,self.n_class))
        for i in range(len(y)):
            matrix[y[i][0],ypred[i][0]] += 1
        return matrix