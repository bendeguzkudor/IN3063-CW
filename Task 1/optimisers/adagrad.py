import numpy as np 

class AdaGrad:
    
    def __init__(self, learning_rate=0.001, epsilon=1e-8):
        # Initialise adagrad optimiser 
        self.learning_rate = learning_rate      # Learning rate 
        self.epsilon = epsilon                  # Small constant to prevent divison by zero 
        self.sg = {}                            # Squared gradient for each parameter

    def update(self, params, grads):

        # Initialise gradients for each parameter
        for p in params:
            if id(p) not in self.sg:
                self.sg[id(p)] = np.zeros_like(p)

        # Loop through parameters and their gradients
        for p, grad in zip(params, grads):
            # get square gradients 
            self.sg[id(p)] += grad**2

            # update computation
            update = -self.learning_rate * grad / (np.sqrt(self.sg[id(p)]) + self.epsilon)

            # apply update to parameter 
            p += update


