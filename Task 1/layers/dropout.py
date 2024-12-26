import numpy as np

class Dropout:
    #Inverted dropout with probability p

    def __init__ (self, p=0.5):
        # Probability of keeping a neuron active
        self.p = p

    def forward(self, x, training = True):
        # Forward pass, if training is true create a scaled binary mask
         if training:
            # Generate mask and scale by 1/p
            self.mask = np.random.binomial(1, self.p, size=x.shape) / self.p
            return x * self.mask
         return x

    def backward(self, grad_output):
        return grad_output * self.mask

