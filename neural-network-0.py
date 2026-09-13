import numpy as np
import pandas as pd

"""
This is my first attempt at building neural network, and I have made almost entire code by myself(except some
 code blocks, which I will mention), and so, for the sake of simplicity, I am making this network with 
only 1 hidden layer as:

        Input Layer(784 paremeters) -------> hidden layer(10 neurons) ----------> output layer(10 activations)

With this model, I asked Gemini to predict the success rate, and Gemini adivisis me to target atleast 80% 

If I complete this project and still have ample time, I will make an optimised version, maybe with less variables
and better success rate by adding one more hidden layer

"""

data = pd.read_csv('mnist_test.csv')
data = np.array(data)
m, n = shape(data)

testing_data = data[0:1000].T
test_Y = testing_data[0]
test_X = testing_data[1:n]

training_data = data[1000:m].T
train_Y = training_data[0]
train_X=  training_data[1:m]

np.random.seed(0)

X = np.random.randn(5, 784).T #taken 5 tests, each of satndard mnist size (784, obv)

def reLU(x):
    return np.maximum(x, 0)

def onehot(Y):                       
    y_onehot = np.zeros((Y.size, Y.max() + 1))      # Tbh, I didn't get how to implement this, so I imported
    y_onehot[np.arange(Y.size), Y] = 1              # it from GeeksForGeeks
    return y_onehot.T

def Initialise():
    w1 = np.random.randn(10, 784) - 0.5
    b1 = np.random.randn(10, 1) - 0.5             #      Used - 0.5 to make sure x*w is small
    w2  = np.random.randn(10, 10) - 0.5           #      and it dose not gets enlarged in the operation
    b2 = np.random.randn(10, 1) - 0.5
    return w1, w2, b1, b2

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))

def Forward(w1, w2, b1, b2, X):     
    z1 = np.dot(w1, X) + b1
    a1 = reLU(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)

def derivative_reLU(z)
    return z > 0                      # In numpy, this condition returns boolean, which is matrix of 1s or 0s
    
def BackPropagation(z1, z2, a1, a2,w1 w2,X, Y):
    m = Y.size
    one_hot_Y = onehot(Y)
    diff_z2 = a2 - one_hot_Y
    d_w2 = (1/m) * np.dot(diff_z2, a1.T)
    d_b2 =  (1/m) * np.sum(diff_z2, axis = 1)

    diff_z1 = np.dot(w2.T, diff_z2) * derivative_reLU(z1)
    d_w1 = (1/m) * dot(diff_z1, X.T)
    d_b1= (1/m) * np.sum(diff_z1, axis = 1)
    pass
