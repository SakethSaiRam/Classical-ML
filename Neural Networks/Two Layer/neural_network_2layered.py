''' Two Layered Neural Network '''

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    
    def __init__(self,lr=0.5,lamda=0.15,num_iters=1000,hidden_layer_size=40,activation='sigmoid'):
        ''' '__init__' takes arguments as Learning Rate(lr), Regularization Constant(lamda), Number of Iterations(num_iters), s1(Hidden Layer Size), func(Type of activation function)'''
        # All these parameters have been initialized by their default values, they can be changed when required
        self.lr = lr
        self.lamda = lamda
        self.num_iters = num_iters
        self.s1 = hidden_layer_size
        self.func = activation

    ''' Sigmoid function maps all the values in the range of 0 to 1 '''
    def sigmoid_function(self,z):
        return 1 / (1 + np.exp(-1 * z))
    
    ''' tanh(z) function maps all values in the range of -1 to 1 '''
    def tanh_function(self,z):
        return np.tanh(z)

    ''' ReLU function maps all negative values to zero and positive values remains same '''
    def reLU_function(self,z):
        return np.maximum(1e-10,z)

    def activation_function(self,z):
        if self.func == 'sigmoid':
            return self.sigmoid_function(z)
        elif self.func == 'relu':
            return self.reLU_function(z)
        elif self.func == 'tanh':
            return self.tanh_function(z)
        else:
            raise ValueError(f"{self.func} Activation is unavailable.")

    ''' Derivates of the activation functions mentioned above '''
    def grad_function(self,a):
        if self.func == 'sigmoid':
            return a * (1 - a)
        elif self.func == 'relu':
            return (a > 1e-10).astype(int)
        elif self.func == 'tanh':
            return (1 - a ** 2)
        else:
            raise ValueError(f"{self.func} Activation is unavailable.")

    ''' Weights initialization '''
    def initialization(self):
        if self.func == 'tanh':
            return np.sqrt(1 / self.s1) # Xavier Initialization
        else:
            return np.sqrt(2 / self.s1) # h-et-al Initialization

    ''' Feature scaling using Standardization Technique '''
    def feature_scale(self,X):
        m,n = X.shape
        for i in range(n):
            X[:,i] = (X[:,i] - np.mean(X[:,i])) / (np.std(X[:,i]) + 1e-18)
        return X
    
    ''' Adds a column of ones at the start as bias and returns the modified array '''
    def add_bias(self,b):
        return np.hstack((np.ones((len(b),1)),b))

    ''' 'fit' function takes X,y and number of different classes as its arguments and trains the neural network '''
    def fit(self,X,y,num_class):
        X = self.feature_scale(X) # Feature Scaling
        X = self.add_bias(X) # Adds bias column
        m,n = X.shape

        self.num_class = num_class

        y_class = np.zeros((m,self.num_class)) # Created a matrix of size m x n_class with all entries as zeros
        
        # Converts 'y' values to 'y_class' as matrix type
        for i in range(m):
            y_class[i][y[i]] = 1

        # Initializing theta vector with random entries for two layers
        ''' If theta's are initialize to zero vector, then the model is no longer a neural network. Since every node do the same computation '''
        self.theta1 = np.random.randn(self.s1, n) * 0.01
        self.theta2 = np.random.randn(num_class, self.s1 + 1) * self.initialization() # Weights Initialization

        # 'Jhistory' and 'iters' keeps track of cost with each iteration
        self.Jhistory = []
        self.iters = []

        for i in range(self.num_iters):
            # Forward Propagation
            a1 = X
            z2 = a1 @ self.theta1.T

            a2 = self.activation_function(z2)
            a2 = self.add_bias(a2)
            z3 = a2 @ self.theta2.T
            a3 = self.activation_function(z3)

            # Cost Function
            part1 = (-1 / m) * np.sum(y_class * np.log(np.abs(a3 + 1e-10)) + (1 - y_class) * np.log(np.abs(1 - a3 + 1e-10)))
            part2 = (self.lamda / (2 * m)) * (np.sum(self.theta1[:, 1:] ** 2) + np.sum(self.theta2[:, 1:] ** 2))
            J = part1 + part2

            self.Jhistory.append(J)
            self.iters.append(i)

            # Back Propagation
            # Finding err(Error), delta, D(gradient).
            err3 = a3 -  y_class # dz3
            err2 = (err3 @ self.theta2) * self.grad_function(a2)
            delta2 = err3.T @ a2
            delta1 = err2[:, 1:].T @ a1
            theta1_temp = self.theta1
            theta1_temp[:, 0] = 0
            theta2_temp = self.theta2
            theta2_temp[:, 0] = 0
            D2 = (1 / m) * (delta2 + self.lamda * theta2_temp)
            D1 = (1 / m) * (delta1 + self.lamda * theta1_temp)

            # Parameter update
            self.theta1 -= self.lr * (D1)
            self.theta2 -= self.lr * (D2)

    ''' Plotting the learning curve between cost and number of iterations '''
    def plot(self,l):
        plt.plot(self.iters,self.Jhistory,color='#00008B',label=l)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost Function(J)")
        plt.title("Cost Function Vs Iterations")
        plt.legend()

    ''' It returns the predicted values given by the trained model '''
    def predict(self,X):
        X = self.feature_scale(X)
        X = self.add_bias(X)
        a1 = X
        z2 = a1 @ self.theta1.T
        a2 = self.activation_function(z2)
        a2 = self.add_bias(a2)
        z3 = a2 @ self.theta2.T
        a3 = self.activation_function(z3)
        y_pred = np.argmax(a3, axis=1).reshape(len(a3), 1)
        return y_pred
    
    ''' It returns the accuracy of the trained model '''
    def accuracy(self,y,ypred):
        return np.mean(y == ypred) * 100