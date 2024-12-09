import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def softmax_backward(dA, Z):
    S = softmax(Z)
    return dA - S
