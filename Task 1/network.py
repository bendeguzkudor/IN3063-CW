import numpy as np
from layers.dense import Dense
from layers.activation import ReLU, Sigmoid, Softmax
from layers.dropout import Dropout

class NeuralNetwork:
    """
    A flexible neural network implementation supporting:
    - Variable layer sizes and depths
    - Multiple activation functions
    - Dropout regularization
    - L1/L2 regularization
    - Different optimizers
    """
    def __init__(self, layer_sizes, activations, optimizer=None, dropout_rate=0.5, 
                 regularization=None, reg_lambda=0.01, seed=42):
        # Store network layers here
        self.layers = []
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.optimizer = optimizer

        # Track parameters for reporting and analysis
        self.training_params = {
            'optimizer': optimizer.__class__.__name__ if optimizer else None,
            'learning_rate': optimizer.learning_rate if optimizer else None,
            'dropout_rate': dropout_rate,
            'regularization': regularization,
            'reg_lambda': reg_lambda,
            'layer_sizes': layer_sizes
        }

        # Set random seed for reproducibility
        np.random.seed(seed)

        # Build network architecture layer by layer
        for i in range(len(layer_sizes)-1):
            # Add dense layer with specified regularization
            self.layers.append(
                Dense(
                    layer_sizes[i], 
                    layer_sizes[i+1],
                    regularization=regularization,
                    reg_lambda=reg_lambda,
                    seed=seed+i  # Unique seed for each layer
                )
            )
            
            # For hidden layers, add activation and dropout
            if i < len(layer_sizes)-2:
                self.layers.append(activations[i])
                self.layers.append(Dropout(p=dropout_rate, seed=seed+i))
            else:
                # Output layer uses Softmax for classification
                self.layers.append(Softmax())

    def forward(self, X, training=True):
        """
        Forward propagation through the network.
        Args:
            X: Input data
            training: Whether to apply dropout (True during training, False during inference)
        """
        current = X
        for layer in self.layers:
            if isinstance(layer, Dropout):
                current = layer.forward(current, training)
            else:
                current = layer.forward(current)
        return current

    def backward(self, grad):
        """
        Backward propagation through the network.
        Updates weights and biases using the optimizer if provided.
        """
        # Collect parameters and gradients for optimization
        params, grads = [], []

        # Gradient clipping for stability
        grad = np.clip(grad, -1, 1)

        # Backward propagation through each layer in reverse
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            
            # Collect parameters and gradients from dense layers
            if isinstance(layer, Dense):
                params.extend([layer.weights, layer.bias])
                grads.extend([layer.weights_grad, layer.bias_grad])
        
        # Update parameters if optimizer is provided
        if self.optimizer:
            self.optimizer.update(params, grads)
        return params, grads
    
    def train_step(self, X, y):
        """
        Single training step including forward pass, loss calculation, and backward pass.
        """
        # Forward pass
        predictions = self.forward(X, training=True)
        
        # Cross-entropy loss calculation with numerical stability
        loss = -np.mean(np.sum(y * np.log(predictions + 1e-8), axis=1))
        
        # Gradient calculation and backward pass
        grad = predictions - y
        self.backward(grad)

        return loss

    def loss(self, y_true, y_pred):
        """Calculate cross-entropy loss between true and predicted values."""
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))