import numpy as np

class Dropout:
    def __init__(self, p=0.5, seed=42):
        # Ensure probability is in valid range [0,1]
        self.p = max(0, min(1, p))
        self.seed = seed
        self.mask = None  # Will store the dropout mask

    def forward(self, x, training=True, seed=None):
        # If not training or no dropout (p=0), return input unchanged
        if not training or self.p == 0:
            return x
        
        # Set random seed for reproducibility
        seed_value = seed if seed is not None else self.seed
        np.random.seed(seed_value)
            
        # Generate and apply dropout mask with scaling factor 1/p
        # This implements "inverted dropout" - scaling during training
        self.mask = np.random.binomial(1, self.p, size=x.shape) / self.p
        return x * self.mask

    def backward(self, grad_output):
        # If no dropout was applied (p=0), pass gradient unchanged
        if self.p == 0:
            return grad_output
            
        # Apply same mask to gradients for consistency
        return grad_output * self.mask