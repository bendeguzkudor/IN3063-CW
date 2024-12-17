import torchvision
from torchvision.transforms import transforms
import torch
import torch.nn.functional as F
import torch.nn as nn

torch.manual_seed(42)



# convert to nparray and normalize 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# get dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


print(train_dataset[0])

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