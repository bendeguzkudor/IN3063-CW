import torchvision
from torchvision.transforms import transforms
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)



# convert to nparray and normalize 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# get dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)



classes = ('airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2)                           #  pooling layer
        self.fc1 = nn.Linear(64 * 8 * 8, 128)                   # fully connected layer
        self.fc2 = nn.Linear(128, 10)                           # output layer 10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # conv2 -> ReLU -> Pool
        x = x.view(-1, 64 * 8 * 8)            # flatten the output
        x = F.relu(self.fc1(x))               # fully connected -> ReLU
        x = self.fc2(x)                       # fully connected -> out
        return x
    
model = CNN()
loss_function = nn.CrossEntropyLoss()
optimizer =optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def train_model(model, train_loader, loss_function, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        for X_train, y_train in train_loader:
            optimizer.zero_grad()            # zero the gradients
            outputs = model(X_train)         # forward pass
            loss = loss_function(outputs, y_train)  #  loss
            loss.backward()                  # backward pass
            optimizer.step()                 # update weights
            train_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}")
    print("Training Complete!")

train_model(model, train_loader, loss_function, optimizer, num_epochs=20)



def evaluate_model(model, test_loader):
    print("\nEvaluating the model...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation
        for X_test, y_test in test_loader:
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)  # Get class index with highest probability
            total += y_test.size(0)
            correct += (predicted == y_test).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

evaluate_model(model, test_loader)