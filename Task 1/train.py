import numpy as np

def train(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    # Prepare a list to track training + validation loss/accuracy
    training_losses = []
    validation_losses = []
    validation_accuracies = []
    # Determine how many mini-batches can be formed from the training data
    n_batches = len(X_train) // batch_size

    for epoch in range(epochs):
        # Shuffle the data indices to ensure different mini-batches each epoch
        perm = np.random.permutation(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

        # Track cumulative loss for the epoch
        epoch_loss = 0

        # Iterate through each mini-batch
        for i in range(n_batches):
            batch_start = i * batch_size
            batch_end = batch_start + batch_size
            
            # Extract features and labels
            X_batch = X_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]
            
            # Update model parameters based on the current batch
            batch_loss = model.train_step(X_batch, y_batch)
            epoch_loss += batch_loss

         # Compute average training loss for the epoch
        avg_train_loss = epoch_loss / n_batches
        training_losses.append(avg_train_loss)

        # Obtain predictions on the validation/test set
        test_pred = model.forward(X_test, training=False)

        # Calculate validation loss (if model.forward provides the loss functionality)
        # Generate predictons for the test set
        test_pred = model.forward(X_test, training=False)

        # Calculate validation loss using predictions and true labels
        val_loss = model.loss(y_test, test_pred)
        validation_losses.append(val_loss)

        # Calculate validation accuracy
        val_accuracy = np.mean(np.argmax(test_pred, axis=1) == np.argmax(y_test, axis=1))
        validation_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Return losses and accuracies for plotting
    return training_losses, validation_losses, validation_accuracies