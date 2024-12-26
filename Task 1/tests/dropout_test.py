import sys
import os
import numpy as np

# Go up one level to Task 1 folder, then into 'layers'
current_dir = os.path.dirname(os.path.abspath(__file__))  
parent_dir = os.path.join(current_dir, "..")             
layers_dir = os.path.join(parent_dir, "layers")           
sys.path.append(layers_dir)

from dropout import Dropout

# Test Dropout with 3x4 array of ones
test_input = np.ones((3, 4)) 
dropout = Dropout(p=0.5)

# Training: Should drop ~50% of values
train_output = dropout.forward(test_input, training=True)
print("Training output:", train_output)

# Test mode: Should all be 1 
test_output = dropout.forward(test_input, training=False)
print("Test output:", test_output)