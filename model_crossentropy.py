import torch as T
import torch.nn as nn


class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)

        y_pred = self.sigmoid(out)
        return y_pred


model = NeuralNet1(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss()


