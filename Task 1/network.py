import numpy as np
from layers.dense import Dense
from layers.activation import ReLU, Sigmoid, Softmax 
from layers.dropout import Dropout

# Checklist
# Layer sizes parameter 
# Units can be changed
# Activation functions parameter
# Adjustable dropout_rate
# Regulariser (L1 or L2)

class NeuralNetwork:
    def __init__(self, layer_sizes, activations)
        # Store network layers here
        self.layers = []

        # Dense layers
        # Activation functions between layers
        # Final layer softmax for classification

    def forward (self, X):
        # Input (X) passes through each layer

        pass

    def backward(self, grad):
        # Calculate gradient for each layer
        # Store parameters and gradients for optimisation

        pass
    
    def train_step(self, X, y):
        # Forward pass -> loss gradient -> backward pass

        pass
        