import torch as T
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

if T.cuda.is_available():
    device = T.device("cuda")


class WineDataset(Dataset):

    def __init__(self, transform=None):
        super(WineDataset, self).__init__()
        xy = np.loadtxt("wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]
        self.num_samples = xy.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.num_samples


def to_tensor(sample):
    inputs, labels = sample
    return T.from_numpy(inputs), T.from_numpy(labels)


class MulTransform():
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, label = sample
        inputs *= self.factor
        return inputs, label

dataset = WineDataset(transform=None)
first_data = dataset[0]
features, labels = first_data
print(first_data)

composed = torchvision.transforms.Compose([to_tensor, MulTransform(2)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(first_data)