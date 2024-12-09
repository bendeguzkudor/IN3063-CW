import numpy as np

# ReLU Function
def relu_forward(x):
    """
    Forward pass for ReLU activation function.
    """
    return np.maximum(0, x)

def relu_backward(dA, x):
    """
    Backward pass for ReLU activation function.
    """
    dZ = np.array(dA, copy=True)
    dZ[x <= 0] = 0
    return dZ

def sigmoid_forward(x):
    """
    Forward pass for sigmoid activation function.
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_backward(dA, x):
    """
    Backward pass for sigmoid activation function.
    """
    sigmoid_output = sigmoid_forward(x)
    return dA * sigmoid_output * (1 - sigmoid_output)

def softmax_forward(logits):
    """
    Compute the softmax probabilities.
    :param logits: Input array of shape (N, D), where N is the batch size and D is the number of classes.
    :return: Softmax probabilities of the same shape as logits.
    """
    max_logits = np.max(logits, axis=1, keepdims=True) 
    exp_logits = np.exp(logits - max_logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def softmax_backward(softmax_output, y_true):
    """
    Compute the gradient of the loss with respect to the input logits.
    Assumes cross-entropy loss.
    :param softmax_output: Softmax probabilities of shape (N, D).
    :param y_true: One-hot encoded true labels of shape (N, D).
    :return: Gradient of the loss with respect to logits.
    """
    batch_size = y_true.shape[0]
    return (softmax_output - y_true) / batch_size
