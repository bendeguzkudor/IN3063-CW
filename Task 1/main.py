import os

# Suppress TensorFlow oneDNN warnings and logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
import numpy as np
from network import NeuralNetwork
from layers.activation import ReLU
from data_loader import load_fashion_mnist
from optimisers.adam import Adam
from optimisers.adagrad import AdaGrad
from train import train
import matplotlib.pyplot as plt

# Using scikit-learn ONLY for evaluation. Need to check if allowed 
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Set the optimiser dynamically
OPTIMISER_NAME = 'adagrad'  # Change to 'adam' for Adam optimiser

# Dynamically create the optimiser
if OPTIMISER_NAME == 'adam':
    optimizer = Adam(learning_rate=0.0001)
elif OPTIMISER_NAME == 'adagrad':
    optimizer = AdaGrad(learning_rate=0.0001)
else:
    raise ValueError(f"Unknown optimiser: {OPTIMISER_NAME}")

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
    optimizer=optimizer,  
    dropout_rate=0.5,            # Dropout rate for regularisation
    regularization=None          # No regularisation for now
)

# Train the model and colelct accuracy over epochs
training_losses, validation_losses, validation_accuracies, training_accuracies = train(
    model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32
)

# Plot Training vs Validation Accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(len(training_accuracies)), training_accuracies, label="Training Accuracy")
plt.plot(range(len(validation_accuracies)), validation_accuracies, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()

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

# Plot Adam's effective learning rate
epochs = range(10)
# Calculate and plot effective learning rate based on optimiser
if isinstance(model.optimizer, Adam):
    effective_lr = [model.optimizer.learning_rate / 
                    (1 - model.optimizer.beta1**(epoch+1)) for epoch in epochs]
elif isinstance(model.optimizer, AdaGrad):
    effective_lr = [model.optimizer.learning_rate for epoch in epochs]
else:
    effective_lr = []  # No effective learning rate for unsupported optimisers

if effective_lr:
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, effective_lr)
    plt.xlabel("Epochs")
    plt.ylabel("Effective Learning Rate")
    plt.title("Effective Learning Rate Over Epochs")
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

