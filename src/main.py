import numpy as np
from activations import relu_forward, relu_backward, sigmoid_forward, sigmoid_backward, softmax_forward, softmax_backward

# Test data

def test_activations():
    # Create test data: 2x3 matrix with positive and negative values 
    x = np.array([[-1, 2, -3], [4, -5, 6]])
    # One-hot labels to test the gradient
    y_true = np.array([[1, 0, 0], [0,1,0]])

    # Now test each activation function

    # ReLU Test: Should output 0 for negative values, keep positive values
    # Expect [0,2,0],[4,0,6] and gradient 0 for negative inputs

    print("ReLU Test:")
    relu_out = relu_forward(x)
    relu_grad = relu_backward(y_true, x)
    print("Input:", x)
    print("ReLU forward:", relu_out)
    print("ReLU backward:", relu_grad)
    print()

if __name__ == "__main__":
    test_activations()
    