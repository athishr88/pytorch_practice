import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import time

start_time = time.time()
device = T.device('cuda' if T.cuda.is_available() else 'cpu')
# device = T.device('cpu')
num_epochs = 10
batch_size = 6
lr = 0.001

# input_size = 28*28
# hidden_layer_size = 100
# num_classes = 10

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                          shuffle=False)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.linear1 = nn.Linear(16*5*5, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        pass


model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

num_steps = len(train_loader)
print(device)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward and loss
        y_pred = model.forward(images)
        loss = criterion(y_pred, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print(f'Epoch = [{epoch+1}/{num_epochs}], Step = [{i+1}/{num_steps}], Loss = {loss.item():.4f}')

    print("Finish Training")

num_samples = 0
num_correct = 0
with T.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        y_pred = model.forward(images)
        _, predictions = T.max(y_pred, 1)

        num_samples += labels.shape[0]
        num_correct += (predictions == labels).sum().item()

    acc = num_correct/num_samples
    print(f'Accuracy: {acc}')

end_time = time.time()
duration = end_time - start_time
print(f'Time evolved = {duration}')
