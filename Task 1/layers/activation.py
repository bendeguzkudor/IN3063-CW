import numpy as np

# ReLU, Sigmoid, and Softmax. Each class has forward and backward methods.

class ReLU:
    def forward(self, input_data):
        # Returns max(0, x) for each element in input_data.
        # Store the input_data for use in the backward pass.
        self.input_data = input_data
        return np.maximum(0, input_data)
       
    def backward(self, grad):
        # Passes the incoming gradient forward only where input_data > 0.
        # If input_data <= 0, gradient is 0.
        
        # Shape debugging
        # print("ReLU backward grad shape:", grad.shape)
        # print("ReLU backward input_data shape:", self.input_data.shape)
        return grad * (self.input_data > 0)


class Sigmoid:
    def forward(self, input_data):
        # Output = 1 / (1 + e^(-x))
        # We store self.output for use in backprop.
        self.output = 1.0 / (1.0 + np.exp(-input_data))
        return self.output
       
    def backward(self, grad):
        # local gradient is self.output * (1 - self.output).

        # print("Sigmoid backward grad shape:", grad.shape)
        return grad * self.output * (1.0 - self.output)
    
class Softmax:
    def forward(self, input_data):
        # Subtract row-wise max for numerical stability..
        max_input = np.max(input_data, axis=1, keepdims=True)
        exp_x = np.exp(input_data - max_input)
        # Normalise by exponential values
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        # print("Softmax forward output shape:", self.output.shape)
        # print("Sample of Softmax output:", self.output[0])
        
        return self.output
       
    def backward(self, grad):
        return grad * self.output * (1.0 - self.output)   