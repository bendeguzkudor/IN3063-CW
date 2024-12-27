import numpy as np

def train(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    # Prepare a list to track accuracy scores across each epoch of training
    accuracies = []

    # Determine how many mini-batches can be formed from the training data
    n_batches = len(X_train) // batch_size

    for epoch in range(epochs):
        # Shuffle the data indices to ensure different mini-batches each epoch
        perm = np.random.permutation(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

        # Iterate through each mini-batch
        for i in range(n_batches):
            batch_start = i * batch_size
            batch_end = batch_start + batch_size
            
            # Extract features and labels
            X_batch = X_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]
            
            # Update model parameters based on the current batch
            predictions = model.train_step(X_batch, y_batch)

        # Obtain predictions on the test set to measure generalisation
        test_pred = model.forward(X_test, training=False)

        # Compare predicted labels to actual labels and calculate the accuracy
        accuracy = np.mean(np.argmax(test_pred, axis=1) == np.argmax(y_test, axis=1))
        accuracies.append(accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.4f}")

    # Return accuracy scores to allow for further analysis 
    return accuracies
