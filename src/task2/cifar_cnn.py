import torchvision
from torchvision.transforms import transforms
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)



# convert to nparray and normalize 
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),        # randomly flip images horizontally
    transforms.RandomCrop(32, padding=4),     # random crop with padding
    transforms.RandomRotation(10),            # random rotation up to 10 degrees
    transforms.ColorJitter(brightness=0.2,    # random changes in brightness, contrast, etc.
                           contrast=0.2, 
                           saturation=0.2, 
                           hue=0.1),
    transforms.ToTensor(),                    # ronvert to tensor
    transforms.Normalize((0.5, 0.5, 0.5),    # ormalize to [-1, 1]
                         (0.5, 0.5, 0.5))
])

# get dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)



classes = ('airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, padding=1) 
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)                           #  pooling layer

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)                   # fully connected layer
        self.fc2 = nn.Linear(256, 10)                           # output layer 10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # conv1 -> ReLU -> Pool
        x = self.dropout1(x)

        x = self.pool(F.relu(self.conv2(x)))  # conv2 -> ReLU -> Pool
        x = self.dropout1(x)

        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)            # flatten the output
        x = F.relu(self.fc1(x))               # fully connected -> ReLU
        x = self.dropout2(x)

        x = self.fc2(x)                       # fully connected -> out
        return x
    
model = CNN()
loss_function = nn.CrossEntropyLoss()
optimizer =optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def train_model(model, train_loader, loss_function, optimizer, num_epochs=5):
    loss_list = []  # Initialize an empty list to store the average loss per epoch
    
    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        for X_train, y_train in train_loader:
            optimizer.zero_grad()            # Zero the gradients
            outputs = model(X_train)         # Forward pass
            loss = loss_function(outputs, y_train)  # Compute loss
            loss.backward()                  # Backward pass
            optimizer.step()                 # Update weights
            train_loss += loss.item()
        
        # Compute and save the average loss for this epoch
        average_loss = train_loss / len(train_loader)
        loss_list.append(average_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")
    
    print("Training Complete!")
    return loss_list  # Return the loss list






# Function to evaluate the model
def evaluate_model(model, train_loader, test_loader, loss_list):
    model.eval()  # Explicitly set to evaluation mode
    
    # Helper function to predict probabilities and labels
    def get_predictions(loader):
        all_labels = []
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for inputs, labels in loader:
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)  # Probabilities
                preds = torch.argmax(probs, dim=1)     # Predicted labels
                
                all_labels.extend(labels.numpy())
                all_preds.extend(preds.numpy())
                all_probs.extend(probs.numpy())
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    # Predict on Train and Test Datasets
    Y_train, Y_train_pred, Y_train_prob = get_predictions(train_loader)
    Y_test, Y_test_pred, Y_test_prob = get_predictions(test_loader)
    
    # Compute Training Metrics
    print("\nModel Performance -")
    print("Training Accuracy:", round(accuracy_score(Y_train, Y_train_pred), 3))
    print("Training Precision (macro):", round(precision_score(Y_train, Y_train_pred, average='macro'), 3))
    print("Training Recall (macro):", round(recall_score(Y_train, Y_train_pred, average='macro'), 3))

    # Compute Validation Metrics
    print("Validation Accuracy:", round(accuracy_score(Y_test, Y_test_pred), 3))
    print("Validation Precision (macro):", round(precision_score(Y_test, Y_test_pred, average='macro'), 3))
    print("Validation Recall (macro):", round(recall_score(Y_test, Y_test_pred, average='macro'), 3))
    
    # Plot the Loss Curve
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_list)
    plt.title('Loss across epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    
    # Plot ROC Curves (One-vs-Rest for Multi-Class)
    plt.subplot(1, 2, 2)
    for i in range(10):  # 10 classes
        fpr, tpr, _ = roc_curve((Y_test == i).astype(int), Y_test_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} AUC = {roc_auc:.2f}')
    
    plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line
    plt.title('ROC Curve for Validation Set')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()



loss_list = train_model(model, train_loader, loss_function, optimizer, num_epochs=15)
evaluate_model(model,train_loader, test_loader, loss_list)


# output

