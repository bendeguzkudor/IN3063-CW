import numpy as np
from tensorflow.keras.datasets import fashion_mnist

def load_fashion_mnist():
    # Get training and test sets from Fashion-MNIST dataset
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    # Flatten the 28x28 images into 784 pixels and normalise pixel values to [0,1] range
    # -1 in reshape calculates the size based on other dimensions
    X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0
    
    # Convert integer labels to one-hot encoded format
    # np.eye creates identity matrix, each row is one class
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    
    return X_train, y_train, X_test, y_test

# Map labels to corresponding clothing items
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']