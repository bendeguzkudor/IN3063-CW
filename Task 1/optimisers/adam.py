import numpy as np

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # Initialise optimiser parameters
        self.learning_rate = learning_rate      # Learning rate
        self.beta1 = beta1                      # Exponential decay rate for first moment
        self.beta2 = beta2                      # Exponential decay rate for second moment
        self.epsilon = epsilon                  # Small constant to prevent division by zero
        self.m = {}                             # First moment vectors
        self.v = {}                             # Second moment vectors
        self.t = 0                              # Timestep counter

    def update(self, params, grads):
        # Initialise moment vectors if first iteration
        if not self.m:
            for p in params:
                self.m[id(p)] = np.zeros_like(p)
                self.v[id(p)] = np.zeros_like(p)

        self.t += 1
        for p, g in zip(params, grads):
            # Update biased first moment estimate
            m = self.beta1 * self.m[id(p)] + (1 - self.beta1) * g
            # Update biased second moment estimate
            v = self.beta2 * self.v[id(p)] + (1 - self.beta2) * g**2
            
            # Compute bias-corrected first and second moment estimates
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            
            # Update parameters using Adam formula
            p -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            # Store moments for next iteration
            self.m[id(p)] = m
            self.v[id(p)] = v
