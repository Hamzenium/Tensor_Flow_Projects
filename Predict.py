import numpy as np
import time
array= np.array([1,2,3,4])

# assert(array.shape == (4,0)
a = np.random.rand(1000000)
b = np.random.rand(1000000)
tic = time.time()
c = np.dot(a,b)
toc = time.time()
d =toc - tic
print(d *1000)
print(array.shape) # shape has column first and then row
# the numpy method is like 300 times faster to calcualte in comaparison
# axsis =0 means vertically, axsis =1 means horizontally
def sigmoid(z):
    s = (1/(1+np.exp(-z)))
    return s


def propagate(w,b,X,Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    cost = (-1/m) * np.sum(Y * np.log(A) + (1-Y) * (np.log(1-A)))
    dw = (1/m) * np.dot(X, (A - Y).T)
    db = (1/m) * np.zum(A-Y)


def optimize(w,b,X,Y,iterations,rate, print_cost = False):
    costs = []
    dw = grads["dw"]
    db = grads["db"]
    for i in range(rate):
        grads, cost  = propagate(w,b,X,Y)
        w = w - rate * dw
        b = b - rate * db
        if i % 100 == 0:
            costs.append(cost)

        return grads, costs

 
def predict(w, b, X):
   
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
         Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    return Y_prediction
