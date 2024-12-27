import os

# Suppress TensorFlow oneDNN warnings and logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
import numpy as np
from network import NeuralNetwork
from layers.activation import ReLU
from data_loader import load_fashion_mnist
from optimisers.adam import Adam
from train import train
import matplotlib.pyplot as plt

# Using scikit-learn ONLY for evaluation. Need to check if allowed 
from sklearn.metrics import classification_report


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

# nEW TESTING FUNCTION
def run_tests(model, X_test, y_test, class_names, num_tests=100):
    correct_count = 0

    for i in range(num_tests):
        # Pick a random index from the test set
        test_idx = np.random.randint(0, len(X_test))
        test_image = X_test[test_idx:test_idx+1]
        actual_class = np.argmax(y_test[test_idx])

        # Generate a prediction from the model
        pred = model.forward(test_image, training=False)
        predicted_class = np.argmax(pred)

        # Compare prediction with actual class
        if predicted_class == actual_class:
            correct_count += 1

        # Log the result (optonal but good for debugging)
        print(f"Test {i+1}: Predicted: {class_names[predicted_class]}, Actual: {class_names[actual_class]}")

    # Calculate and display accuracy percentage
    accuracy = (correct_count / num_tests) * 100
    print(f"\nAccuracy over {num_tests} tests: {accuracy:.2f}%")

# Run the tests (img display removed)
run_tests(model, X_test, y_test, class_names, num_tests=1000)

# Collect true labels and predictions
y_pred = [np.argmax(model.forward(X_test[i:i+1], training=False)) for i in range(len(X_test))]
y_true = [np.argmax(y) for y in y_test]

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Adam tests
# 100 tests returned 85% accuracy
# 2nd attempt at 100 tests, 85% accuracy
# 1000 tests, 83% accuracy
