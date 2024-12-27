import numpy as np
from network import NeuralNetwork
from layers.activation import ReLU
from data_loader import load_fashion_mnist
from optimisers.adam import Adam
from train import train
import matplotlib.pyplot as plt
import os

# Suppress TensorFlow oneDNN warnings and logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs

# Now import TensorFlow and other modules
import tensorflow as tf
# Other imports and your code...

# Load the Fashoin-MNIST dataset
X_train, y_train, X_test, y_test = load_fashion_mnist()

# Create a neural netwrok model
# with specified layer sizes and ReLU activation
model = NeuralNetwork(
    layer_sizes=[784, 256, 10],   # 784 input dims, 256 hidden, 10 output
    activations=[ReLU()],        # Use ReLU for hidden layer
    optimizer=Adam(learning_rate=0.0001)  # Adam optimser with chosen LR
)

# Train the model and colelct accuracy over epochs
accuracies = train(model, X_train, y_train, X_test, y_test, epochs=5)

# Class names for Fashion-MNIST labels
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Evaluate the model on random samples from the test set
# TODO: Prevent image from displaying, greatly increase no. of tests, give % accuracy
for i in range(10):
    # Pick a random index from the test set
    test_idx = np.random.randint(0, len(X_test))
    test_image = X_test[test_idx:test_idx+1]

    # Generate a prediction from the model
    pred = model.forward(test_image, training=False)
    predicted_class = np.argmax(pred)
    actual_class = np.argmax(y_test[test_idx])

    # Print out predicted vs. actual class
    print(f"\nTest {i+1} Results:")
    print(f"Predicted: {class_names[predicted_class]}")
    print(f"Actual: {class_names[actual_class]}")

    # Show the test image and the predicted label
    plt.imshow(test_image.reshape(28,28), cmap='gray')
    plt.title(f"Prediction: {class_names[predicted_class]}")
    plt.show()
