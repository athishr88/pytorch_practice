import torch as T
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

# MNIST
# Dataloader, Transformation
# Multilayer Neural network and activations
# Loss and optimizer
# Training loop (Batch training)
# Model evaluation
# GPU support
start_time = time.time()

# device = T.device('cuda' if T.cuda.is_available() else 'cpu')
device = T.device('cpu')
# Hyper parameters
input_size = 28*28
hidden_layer_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                              transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                              transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                           shuffle=False)

# examples = iter(train_loader)
# data, labels = examples.next()
# print(data.shape, labels.shape)

# plt.figure()
# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.imshow(data[i][0], cmap='gray')
#
# plt.show()

# Setup NN
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        y_pred = self.linear2(out)
        return y_pred


model = NeuralNet(input_size, hidden_layer_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = T.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
num_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        layer = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)

        # Forward pass
        y_pred = model.forward(layer)
        loss = criterion(y_pred, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoc: {epoch+1}/{num_epochs}, step: {i+1}/{num_steps}, loss: {loss.item():.4f}')

# Test
with T.no_grad():
    n_correct = 0
    n_samples = 0
    for i, (images, labels) in enumerate(test_loader):
        layer = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)

        # Prediction
        y_pred = model.forward(layer)
        loss = criterion(y_pred, labels)

        _, predictions = T.max(y_pred, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100*(n_correct/n_samples)
    print(f'{n_samples} Accuracy = {acc}')

end_time = time.time()
duration = end_time - start_time
print(f'Time evolved = {duration}')

