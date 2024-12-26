import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../')  # Go up one level to Task 1 folder to find data_loader.py
from data_loader import load_fashion_mnist

X_train, y_train, X_test, y_test = load_fashion_mnist()

# Data training shape should be (60000, 784)
# 60,000 images, 784 shape (28x28 pixels flattened)
# Data training labels shape (60000, 10), 10 labels
print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)

# Should be (10,000, 784)
print("Testing data shape:", X_test.shape)
# Should be (10,000, 10)    
print("Testing labels shape:", y_test.shape)  

# Image visualisation test
plt.imshow(X_train[0].reshape(28,28), cmap='gray')
plt.show()
