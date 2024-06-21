''' Multilayered Neural Network '''

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:

    def __init__(self,lr=0.5,lamda=0.15,num_iters=50,hidden_layer_sizes=[16,16,16,16],batch_size=500,activation='sigmoid'):
        ''' '__init__' takes arguments as Learning Rate(lr), Regularization Constant(lamda), Number of Iterations(num_iters), layers(Hidden Layer Size as a list), Batch Size, func(Type of activation function) '''
        # All these parameters have been initialized by their default values, they can be changed when required
        self.lr = lr
        self.lamda = lamda
        self.num_iters = num_iters
        self.layers = hidden_layer_sizes
        self.bs = batch_size
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
    def initialization(self,size):
        if self.func == 'tanh':
            return np.sqrt(1 / size) # Xavier Initialization
        else:
            return np.sqrt(2 / size) # h-et-al Initialization

    ''' Feature scaling using Standardization Technique '''
    def feature_scale(self,X):
        m,n = X.shape
        for i in range(n):
            X[:,i] = (X[:,i] - np.mean(X[:,i])) / (np.std(X[:,i]) + 1e-18)
        return X
    
    ''' Adds a column of ones at the start as bias and returns the modified array '''
    def add_bias(self,b):
        return np.hstack((np.ones((len(b),1)),b))

    ''' Removes the first column from the given array and returns the modified array '''
    def remove_bias(self,b):
        return b[:,1:]

    ''' 'fit' function takes X,y and number of different classes as its arguments and trains the neuarl network '''
    def fit(self,X,y,num_class):
        X = self.feature_scale(X) # Feature scaling
        X = self.add_bias(X) # Adds bias column
        m,n = X.shape
        
        # Adding input and output layer dimensions
        self.layers = [n] + self.layers + [num_class]
        L = len(self.layers)

        self.num_class = num_class

        # Converts 'y' values to 'y_class' as matrix type
        y_class = np.zeros((m,self.num_class))
        for i in range(m):
            y_class[i][y[i]] = 1

        # 'Jhistory' and 'iters' keeps track of cost with each iteration
        self.Jhistory = []
        self.iters = []

        it = 0 # Stores the iteration value
        self.theta = []

        # Initializing parameter vectors theta.
        for i in range(1,L):
            if i == 1:
                self.theta.append(np.random.randn(self.layers[i],self.layers[i - 1]) * 0.01)
            else:
                self.theta.append(np.random.randn(self.layers[i],self.layers[i - 1] + 1) * self.initialization(self.layers[i - 1]))
         
        for i in range(self.num_iters):
            for k in range(m // self.bs):
                X_mini = X[k*self.bs:(k+1)*self.bs,:]
                y_mini = y_class[k*self.bs:(k+1)*self.bs,:]
                
                # Forward Propagation
                a = []
                for j in range(L-1):
                    if j == 0:
                        a1 = X_mini
                        a.append(a1)
                        z2 = a1 @ self.theta[j].T
                        a2 = self.activation_function(z2)
                        a2 = self.add_bias(a2)
                    elif j < (L -2):
                        a1 = a2
                        z2 = a1 @ self.theta[j].T
                        a2 = self.activation_function(z2)
                        a2 = self.add_bias(a2)
                    else:
                        a1 = a2
                        z2 = a1 @ self.theta[j].T
                        a2 = self.sigmoid_function(z2)
                    a.append(a2)

                # Cost Function
                J = (-1 / self.bs) * np.sum(y_mini * np.log((a[-1]+1e-10)) + (1 - y_mini) * np.log((1-a[-1]+1e-10)))
                for j in range(L-1):
                    J += (self.lamda / (2 * self.bs)) * (np.sum((self.theta[j])[:,1:] ** 2))
                self.Jhistory.append(J)
                it += 1
                self.iters.append(it)

                # Back Propagation
                delta = []
                grad = []
                for j in range(L-1,0,-1):
                    if j == (L - 1):
                        delta2 = a[j] - y_mini
                        delta_temp = delta2.T @ a[j-1]
                    elif j == (L - 2):
                        delta2 = (delta1 @ self.theta[j]) * self.grad_function(a[j])
                        delta_temp = delta2[:,1:].T@a[j-2]
                    else:
                        delta1 = delta1[:,1:]
                        delta2 = (delta1 @ self.theta[j]) * self.grad_function(a[j])
                        delta_temp = delta2[:,1:].T @ a[j - 1]

                    theta_temp = self.theta[j - 1]
                    theta_temp[:,0] = 0
                    grad.insert(0, ((1 / self.bs) * (delta_temp + self.lamda * theta_temp)))
                    delta1 = delta2
                
                # Parameter update
                for j in range(L-1):
                    self.theta[j] -= self.lr*(grad[j])
    
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
        L = len(self.layers)
        for j in range(L-1):
            if j == 0:
                a1 = X
                z2 = a1 @ self.theta[j].T
                a2 = self.activation_function(z2)
                a2 = self.add_bias(a2)
            elif j < (L -2):
                a1 = a2
                z2 = a1 @ self.theta[j].T
                a2 = self.activation_function(z2)
                a2 = self.add_bias(a2)
            else:
                a1 = a2
                z2 = a1 @ self.theta[j].T
                a2 = self.sigmoid_function(z2)
        ypred = np.argmax(a2,axis=1).reshape(len(a2),1)
        return ypred
    
    ''' It returns the accuracy of the trained model '''
    def accuracy(self,y,ypred):
        return np.mean(y == ypred) * 100