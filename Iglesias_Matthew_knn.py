import numpy as np
import time

class knn(object):
    def __init__(self,k=3,weighted=True,classify=True, distance = 'Euclidean'):  
        self.k = k
        self.weighted = weighted
        self.classify = classify
        self.distance = distance
    
    def euclidean_distance(self,X_test):
        # Returns 2D array dist
        # where dist[i,j] is the Euclidean distance from training example i to test example j
        dist = np.sum(self.X_train**2,axis=1).reshape(-1,1) # dist = X_train**2
        dist = dist - 2*np.matmul(self.X_train,X_test.T) # dist = X_train**2  - 2*X_train*X_test
        dist = dist + np.sum(X_test.T**2,axis=0).reshape(1,-1) # dist = X_train**2  - 2*X_train*X_test + X_test**2 - Not really necessary
        dist = np.sqrt(dist) 
        return  dist
    
    def manhattan_distance(self,X_test):
        # Returns 2D array dist
        # where dist[i,j] is the Manhattan distance from training example i to test example j
        # Your code goes here
        dist = np.sum(self.X_train ** 2, axis=1).reshape(-1,1)
        dist = dist - 2*np.matmul(self.X_train,X_test.T) # dist = X_train**2  - 2*X_train*X_test
        dist = dist + np.sum(X_test.T**2,axis=0).reshape(1,-1) # dist = X_train**2  - 2*X_train*X_test + X_test**2 - Not really necessary
        dist = np.abs(dist)
        return dist
    
    def fit(self,x,y):
        # K-nn just stores the training data
        self.X_train = x
        self.y_train = y
    
    def predict(self,x_test):
        if self.distance=='Euclidean':
            dist  = self.euclidean_distance(x_test)     
        else:
            dist  = self.manhattan_distance(x_test)     
        
        nn = np.argsort(dist,axis=0)[:self.k]
        dist = np.sort(dist,axis=0)[:self.k]
        ind = np.arange(len(y_test))
        if self.weighted:
            w = 1/(dist**2+1e-10)
            sum_w = np.sum(w,axis=0)
            w = w/sum_w
        else:
            w = np.zeros_like(nn,dtype=np.float32)+1/self.k
        if self.classify:
            vote = np.zeros((len(y_test),np.max(self.y_train)+1))
            for i in range(self.k):
                vote[ind,self.y_train[nn[i,:]]] += w[i,:]
            pred = np.argmax(vote,axis=1)
        else:
            pred = np.sum(self.y_train[nn]*w,axis=0)
        return pred
    
    def root_kd(X):
        #returns 378 root example from KD tree first find the standard deviation and median
        s = np.std(X,axis=0)
        highest_sd = np.argmax(s)
        med_sd = np.median(highest_sd)
        return med_sd
    
    def nn_graph(X,k):
        #returns k nearest neighbor graph of dataset x
        #retuns x.shape[0] by k array of ints where elements in row i are indecies of NN of ex i in dataset
        if k == 1:
            k_i = X.argmin(0)
        else:
            k_i = X.argpartition(k, axis = 0)[:k]
        k_d = np.take_along_axis(X, k_i, axis = 0) 
        sorted_idx = k_d.argsort(axis = 0)
        k_i_sorted = np.take_along_axis(k_i, sorted_idx, axis = 0)
        return k_i_sorted
   
def split_train_test(X,y,percent_train=0.9):
    ind = np.random.permutation(X.shape[0])
    train = ind[:int(X.shape[0]*percent_train)]
    test = ind[int(X.shape[0]*percent_train):]
    return X[train],X[test], y[train], y[test]    
    
if __name__ == "__main__":  
    print('MNIST dataset')
   
    X = np.load('mnist_X.npy').astype(np.float32).reshape(-1,28*28)
    y = np.load('mnist_y.npy')
    
    n = X.shape[0] # Use all examples
    #n = 10000      # Use a few examples
    
    ind = np.random.permutation(len(y))
    X=X[ind[:n]]
    y=y[ind[:n]]
    
    X_train, X_test, y_train, y_test = split_train_test(X,y)
    
    model = knn()
    start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time()-start
    
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    
    start = time.time()       
    pred = model.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    print('Accuracy:',np.sum(pred==y_test)/len(y_test))
    
    model = knn(weighted=False)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time()-start
    
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    
    start = time.time()       
    pred = model.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    print('Accuracy:',np.sum(pred==y_test)/len(y_test))
    
    