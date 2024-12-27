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
    def __init__(self, layer_sizes, activations, optimizer=None, dropout_rate=0.5, regularization=None, reg_lambda=0.01):
        # Store network layers here
        self.layers = []
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.optimizer = optimizer

        # Track parameters to help with report 
        self.training_params = {
            'optimizer': optimizer.__class__.__name__ if optimizer else None,
            'learning_rate': optimizer.learning_rate if optimizer else None,
            'dropout_rate': dropout_rate,
            'regularization': regularization,
            'reg_lambda': reg_lambda,
            'layer_sizes': layer_sizes
        }

        # Build network architecture by adding layers one by one
        for i in range(len(layer_sizes)-1):
            # Add a dense layer with regularization
            self.layers.append(
                Dense(
                    layer_sizes[i], 
                    layer_sizes[i+1],
                    regularization = regularization,
                    reg_lambda = reg_lambda
                    )
                )
            
            # Add appropriate activation function
            # For the output layer (last layer), we use Softmax for classification
            # For hidden layers, we use the activation function provided
            # Add activation and dropout for hidden layers
            if i < len(layer_sizes)-2:
                self.layers.append(activations[i])
                # Add dropout layer with specified rate
                self.layers.append(Dropout(p=dropout_rate))
            else:
                self.layers.append(Softmax())


    def forward(self, X, training=True):
        # Forward propagation with dropout control
        current = X
        for layer in self.layers:
            # Apply dropout only during training
            if isinstance(layer, Dropout):
                current = layer.forward(current, training)
            else:
                current = layer.forward(current)
        return current

    def backward(self, grad):
        # Calculate gradient for each layer
        # Store parameters and gradients for optimisation
        params, grads = [], []

        # Gradient clipping for stability
        grad = np.clip(grad, -1, 1)

        #Backward propagation through the network, in reverse order
        for layer in reversed(self.layers):
            # Propagate the gradient through current layer
            grad = layer.backward(grad)
            # Collect parameters and gradients from dense layers only
            if isinstance(layer, Dense):
                params.extend([layer.weights, layer.bias])
                grads.extend([layer.weights_grad, layer.bias_grad])
        
        # Update the optimizer if available
        if self.optimizer:
            self.optimizer.update(params, grads)
        return params, grads
    
    def train_step(self, X, y):
        # Forward pass -> loss gradient -> backward pass
        predictions = self.forward(X, training = True)
        grad = predictions - y
        params, grads = self.backward(grad)

        return predictions
