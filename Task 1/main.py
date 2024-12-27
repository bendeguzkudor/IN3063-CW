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
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# Load the Fashoin-MNIST dataset
X_train, y_train, X_test, y_test = load_fashion_mnist()

# Class names for Fashion-MNIST labels
class_names = [ 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Create a neural netwrok model
# with specified layer sizes and ReLU activation
model = NeuralNetwork(
    layer_sizes=[784, 256, 10],   # 784 input dims, 256 hidden, 10 output
    activations=[ReLU()],        # Use ReLU for hidden layer
    optimizer=Adam(learning_rate=0.0001),  # Adam optimser with chosen LR
    dropout_rate=0.5,            # Dropout rate for regularisation
    regularization=None          # No regularisation for now
)

# Train the model and colelct accuracy over epochs
training_losses, validation_losses, validation_accuracies = train(
    model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32
)


# Plot Training vs Validation Loss
plt.plot(range(len(training_losses)), training_losses, label="Training Loss")
plt.plot(range(len(validation_losses)), validation_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()

# Plot Validation Accuracy
plt.plot(range(len(validation_accuracies)), validation_accuracies, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Validation Accuracy Over Epochs")
plt.show()

# Evaluate the model on the test set
y_pred = [np.argmax(model.forward(X_test[i:i+1], training=False)) for i in range(len(X_test))]
y_true = [np.argmax(label) for label in y_test]

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Generate and display confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Calculate and display per-class accuracy
class_accuracies = cm.diagonal() / cm.sum(axis=1)
for i, accuracy in enumerate(class_accuracies):
    print(f"Accuracy for {class_names[i]}: {accuracy:.2%}")

"""
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


# Adam tests
# 100 tests returned 85% accuracy
# 2nd attempt at 100 tests, 85% accuracy
# 1000 tests, 83% accuracy
"""