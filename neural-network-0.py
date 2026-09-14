import numpy as np
import pandas as pd
import os

"""
This is my first attempt at building neural network, and I have made almost entire code by myself(except some
 code blocks, which I will mention), and so, for the sake of simplicity, I am making this network with 
only 1 hidden layer as:

        Input Layer(784 paremeters) -------> hidden layer(64 neurons) ----------> output layer(10 activations)

With this model, I asked Gemini to predict the success rate, and Gemini adivisis me to target atleast 80% 

If I complete this project and still have ample time, I will make an optimised version, maybe with less variables
and better success rate by adding one more hidden layer

"""

def load_csv(file_path):
    data = pd.read_csv(file_path)
    data = np.array(data)

    testing_data = data[0:1000].T
    test_Y = testing_data[0]
    test_X = testing_data[1:785]
    test_X = test_X/255

    training_data = data[1000:].T
    train_Y = training_data[0]
    train_X=  training_data[1:785]
    train_X = train_X/255

    return train_X, train_Y, test_X, test_Y

np.random.seed(0)

def Initialise():
    W1 = np.random.randn(64, 784) * 0.1
    b1 = np.random.randn(64, 1) * 0.1             #      Used * 0.1 to make sure x*w is small
    W2 = np.random.randn(10, 64) * 0.1            #      and it dose not gets enlarged in the operation
    b2 = np.random.randn(10, 1) * 0.1
    return W1, b1, W2, b2

def InitialiseOrLoad(filename='trained_weights.npz'):
    if os.path.exists(filename):
        print("Saved data from previous training found! Loading...")
        saved_data = np.load(filename)
    
        W1 = saved_data['W1']
        b1 = saved_data['b1']
        W2 = saved_data['W2']
        b2 = saved_data['b2']

    else:
        print("No trained data found, initialising random weights and biases...")
        W1, b1, W2, b2 = Initialise()

    return W1, b1, W2, b2

def reLU(x):
    return np.maximum(x, 0)

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))         # tbh, i didn't know how to code this, so I imported
    one_hot_Y[np.arange(Y.size), Y] = 1                 # it from geeksforgeeks o_o
    return one_hot_Y.T


def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


def Forward(W1, b1, W2, b2, X):

    # Layer 1 (Hideen)     
    z1 = np.dot(W1, X) + b1
    a1 = reLU(z1)

    # Layer 2 (Output)
    z2 = np.dot(W2, a1) + b2
    a2 = softmax(z2)

    return z1, a1, z2, a2


def derivative_reLU(z):
    return z > 0                # In numpy, this condition returns boolean, which is matrix of 1s or 0s
    
def BackPropagation(z1, a1, z2, a2, W1, W2, X, Y):     # mention to stackExchange for helping me out 
    m = Y.size                                         # with these formulas for higher order matrices
    one_hot_Y = one_hot(Y)
    dz2 = a2 - one_hot_Y
    dW2 = (1/m) * np.dot(dz2, a1.T)
    db2 =  (1/m) * np.sum(dz2, axis = 1, keepdims = True)

    dz1 = np.dot(W2.T, dz2) * derivative_reLU(z1)
    dw1 = (1/m) * np.dot(dz1, X.T)
    db1= (1/m) * np.sum(dz1, axis = 1, keepdims  = True)
    
    return dw1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dw1, db1, dW2, db2, learning_rate):
    W1 = W1 - learning_rate * dw1
    W2 = W2 - learning_rate * dW2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2

    return W1, b1, W2, b2

def predict(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y)/Y.size

def gradient_descent(X, Y, learning_rate, iterations):
    W1, b1, W2, b2 = InitialiseOrLoad()
    
    for i in range(iterations + 1):
        Z1, A1, Z2, A2 = Forward(W1, b1, W2, b2, X)

        dw1, db1, dW2, db2 = BackPropagation(Z1, A1, Z2, A2, W1, W2, X, Y)

        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dw1, db1, dW2, db2, learning_rate)

        if i%50 == 0:
            predictions = predict(A2)
            accuracy = get_accuracy(predictions, Y)
            print(f"Iteration: {i} | Accuracy: {accuracy * 100:.2f}%")
        
    return W1, b1, W2, b2

if __name__ == "__main__":
    print("Loading data...")
    train_X, train_Y, test_X, test_Y = load_csv('mnist_test.csv')
    
    print("Starting training...")
    W1, b1, W2, b2 = gradient_descent(train_X, train_Y, learning_rate = 0.1, iterations = 500)

    np.savez('trained_weights.npz', W1=W1, b1=b1, W2=W2, b2=b2)
    print("Network memory saved successfully!")

