''' K-Nearest Algorithm '''

# Importing numpy for numerical calculations
import numpy as np

class KNearestNeighbours:

    def euclideanDistance(self,p1,p2):
        ''' It calculates the euclidean distance between two instances and returns the distace '''
        return np.sqrt(np.sum((p1 - p2) ** 2, axis = 1))

    def fit(self,X_train,y_train):
        ''' 'fit' method takes X_train, y_train (Data to get neighbouring values) as arguments.'''
        self.X_train = X_train
        self.y_train = y_train

    def mode(self,l):
        ''' 'mode' method takes and l(list of elements) and returns most frequently occurred element.
        If multiple elements occur same number of times then smallest among them is returned.'''
        return list(max(l,key=l.count))
    
    def predict(self,X_test,k):
        ''' It takes X_test(Test data point), k(Numbers of neighbours to be considered) and returns predicted values.'''
        
        m,n = self.X_train.shape
        mt,nt = X_test.shape

        ypred = [] # Initializing y_pred(list) to store predicted values.

        for i in range(mt):
            test_point = X_test[i,:]
            # Calculation of distance between training set and test case.
            dists = self.euclideanDistance(self.X_train,test_point)
            # Stores indices of distance values in the increasing order of distance.
            sorted_dists_indices = np.argsort(dists)
            # Stores y values of k nearest neighbours as list.
            k_nearest_neighbours = list((self.y_train[sorted_dists_indices])[:k])
            # Finds most frequent neighbour and assigns its value as ypred(predicted y).
            ypred.append(self.mode(k_nearest_neighbours))

        ypred = np.array(ypred) # Converting the list in to array
        return ypred

    def accuracy(self,y,ypred):
        ''' It returns the accuracy of the testdata '''
        return (np.mean(y == ypred) * 100)