''' Linear Regression Model '''

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    # Initializing some useful variables
    def __init__(self,lr = 0.05,lamda = 0.0001,L1_ratio=0.5,num_iters = 500):
        ''' '__init__' takes arguments as Learning Rate(lr), Regularization Constant(lamda), L1 ratio, Number of Iterations(num_iters)'''
        # All these parameters have been initialized by their default values, they can be changed when required
        self.lr = lr
        self.lamda = lamda
        self.L1_ratio = L1_ratio
        self.num_iters = num_iters

    ''' Feature scaling using Standardization Technique '''
    def feature_scale(self,X):
        for i in range(X.shape[1]):
            X[:,i] = (X[:,i] - np.mean(X[:,i]))/(np.std(X[:,i] + 1e-8)) 
        return X
    
    ''' It takes X and theta as its arguments and returns Hypothesis Value(h = Matrix Multiplication of X and theta) '''
    def hypothesis_function(self,X,theta):
        return X @ theta
    
    ''' Cost method takes X,y and theta as its arguments and returns cost value with regularization '''
    def cost_function(self,X,y,theta):
        # Elastic Net Regression have been used here, L1_ratio decides the type of regularization(L1 or L2)
        m,n = X.shape
        h = self.hypothesis_function(X,theta)
        part1 = (1 / (2 * m)) * np.sum((y - h) ** 2)
        part2 = np.abs((self.lamda) * (self.L1_ratio) * np.sum(theta[1:,:]))
        part3 = (self.lamda) * ((1 - self.L1_ratio) / 2) * np.sum((theta[1:,:]) ** 2)
        J =  part1 + part2 + part3
        return J

    ''' 'fit' function takes X,y as its arguments and applies Gradient Descent Algorithm to get theta '''
    def fit(self,X,y):
        X = self.feature_scale(X) # Feature Scaling
        X = np.hstack((np.ones((len(X),1)),X)) # Adds Bias column
        
        m,n = X.shape
        # Initializing theta with all entries as zeros
        self.theta = np.zeros((n,1))

        # 'Jhistory' and 'iters' keeps track of cost with each iteration
        self.Jhistory,self.iters = [],[]

        # Applying Gradient Descent Algorithm
        for i in range(self.num_iters):
            theta_temp = self.theta
            theta_temp[1:,:] = 0

            h = self.hypothesis_function(X,self.theta)
            dp1 = (self.lr / m) * ((X.T) @ (h - y))
            dp2 = (self.lr) * self.lamda * self.L1_ratio
            dp3 = (self.lr) * self.lamda * (1 - self.L1_ratio) * theta_temp
            self.theta -= (dp1 + dp2 + dp3)

            # Storing the value of Cost(J) in each iteration
            self.Jhistory.append(self.cost_function(X,y,self.theta))
            self.iters.append(i)

    ''' Plotting the learning curve between cost and number of iterations '''
    def plot(self):
        plt.plot(self.iters,self.Jhistory,color='#00008B')
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost Function(J)")
        plt.title("Cost Function Vs Iterations")

    ''' Computes R2 Square(Co-efficient of Determination) '''
    def r2score(self,y,ypred):
        score = 1 - ((y - ypred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        return score

    ''' It returns the predicted values given by the trained model '''
    def predict(self,X):
        X = self.feature_scale(X)
        X = np.hstack((np.ones((len(X),1)),X)) # Adds Bias column
        return X @ self.theta

    ''' It returns the accuracy of the trained model '''
    def accuracy(self,y,ypred):
        return np.mean(y == ypred) * 100

    ''' It returns the final predicted values with respect to the given threshod value '''
    def ypred_threshold(self,ypred,thresh):
        return np.ceil(ypred - thresh)

    ''' It returns the threshold and maximum training accuracy. '''
    def threshold(self,y,ypred):
        thresh_values = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

        accscore_values = [] # Stores accuracy values for each threshold

        for i in thresh_values:
            new_ypred = self.ypred_threshold(ypred,i)
            accscore_values.append(self.accuracy(y,new_ypred))

        index_maxacc = np.argmax(accscore_values)
        max_accscore = accscore_values[index_maxacc]
        threshold_value = thresh_values[index_maxacc]

        return threshold_value,max_accscore