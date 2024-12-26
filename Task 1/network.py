import numpy as np
from layers.dense import Dense
from layers.activation import ReLU, Sigmoid, Softmax 
from layers.dropout import Dropout

# Checklist
# Layer sizes parameter DONE
# Units can be changed DONE
# Activation functions parameter DONE
# Adjustable dropout_rate
# Regulariser (L1 or L2)

class NeuralNetwork:
    def __init__(self, layer_sizes, activations)
        # Store network layers here
        self.layers = []

 # Build network architecture by adding layers one by one
        for i in range(len(layer_sizes)-1):
            # Add a dense layer connecting current layer size to next layer size
            self.layers.append(Dense(layer_sizes[i], layer_sizes[i+1]))
            
            # Add appropriate activation function
            # For the output layer (last layer), we use Softmax for classification
            # For hidden layers, we use the activation function provided
            if i < len(layer_sizes)-2:
                self.layers.append(activations[i])
            else:
                self.layers.append(Softmax())


    def forward (self, X):
        current = x
        # Input X through each layer in order
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def backward(self, grad):
        # Calculate gradient for each layer
        # Store parameters and gradients for optimisation
        params, grads = [], []

        #Backward propagation through the network, in reverse order
        for layer in reversed(self.layers):
            # Propagate the gradient through current layer
            grad = layer.backward(grad)
            # Collect parameters and gradients from dense layers only
            if isinstance(layer, Dense):
                params.extend([layer.weights, layer.bias])
                grads.extend([layer.weights_grad, layer.bias_grad])
        return params, grads
    
    def train_step(self, X, y):
        # Forward pass -> loss gradient -> backward pass
        predictions = self.forward(X)
        grad = predictions - y
        params, grads = self.backward(grad)

        return predictions
