import numpy as np

class Dense:
    def __init__(self, input_size, output_size, regularization = None, reg_lambda = 0.01, seed=42):
        np.random.seed(seed)
        self.input_size = input_size
        self.output_size = output_size
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        # Initialising weights with small random values (scaled by 0.01)
        # Bias is set to zero for simplicity
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        
    def forward(self, input):
        # Forward pass for the Dense layer.
        # Calculates the output of the layer using: input * weights + bias

        self.input = input  # Store the input for later use in backward pass
        return np.dot(input, self.weights) + self.bias 
        
    def backward(self, grad):
        # Backward pass for the Dense layer.
        # Calculates gradients for weights and bias based on the loss gradient (grad)
        # Returns the gradient of the loss with respect to the input (for propagating backward)
        # Gradient of the weights: input^T * grad
        self.weights_grad = np.dot(self.input.T, grad)
        # Gradient of the bias: sum of grad across all inputs
        self.bias_grad = np.sum(grad, axis=0, keepdims=True)
        # Gradient of the loss wrt the input: grad * weights^T
        return np.dot(grad, self.weights.T)

# TEST
# (1,3) input
"""
dense = Dense(input_size =3, output_size = 2)
test_input = np.array([[1, 2, 3]])
output = dense.forward(test_input)
#Print should be (1, 2)
print("Dense layer output shapre:", output.shape)
"""