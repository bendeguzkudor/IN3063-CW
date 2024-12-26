import sys
import os
import numpy as np

# Go up one level to Task 1 folder, then into 'layers'
#TODO: Create an __init__.py and turn Task 1 to task1 package to clean up tests
current_dir = os.path.dirname(os.path.abspath(__file__))  # e.g. .../Task 1/tests
parent_dir = os.path.join(current_dir, "..")              # e.g. .../Task 1
layers_dir = os.path.join(parent_dir, "layers")           # e.g. .../Task 1/layers
sys.path.append(layers_dir)

from activation import ReLU, Sigmoid  # now Python finds activation.py

# Test data
test_input = np.array([-2, -1, 0, 1, 2])

# Test ReLU
relu = ReLU()
relu_output = relu.forward(test_input)
# Expected to turn all negative values into 0
print("ReLU output:", relu_output)  


# Test Sigmoid
sigmoid = Sigmoid()
sigmoid_output = sigmoid.forward(test_input)
# Expected to turn all values into values between 0 and 1 (sig)
print("Sigmoid output:", sigmoid_output)  

