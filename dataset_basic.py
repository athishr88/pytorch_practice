import torch as T
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

if T.cuda.is_available():
    device = T.device("cuda")


class WineDataset(Dataset):

    def __init__(self):
        super(WineDataset, self).__init__()
        xy = np.loadtxt("wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.x = T.from_numpy(xy[:, 1:])
        self.y = T.from_numpy(xy[:, [0]])
        self.num_samples = xy.shape[0]
        # self.num_features = xy[0].shape[1]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.num_samples


dataset = WineDataset()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

# dataiter = iter(dataloader)
# data = dataiter.next()
# features, label = data
# print(data)

# Training loop
num_epochs = 2
batch_size = 4
total_samples = len(dataset)
num_iterations = math.ceil(total_samples / batch_size)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        if (i+1) % 5 == 0:
            print(f'epoch: {epoch+1}/{num_epochs}, step: {i+1}/{num_iterations}, inputs: {inputs.shape}')
