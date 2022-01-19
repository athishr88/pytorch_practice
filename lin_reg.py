import torch as T
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import datasets

X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = T.from_numpy(X_numpy.astype(np.float32))
y = T.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

# 1) Model

model = nn.Linear(n_features, n_features)

# 2) Loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = T.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
no_epochs = 1000

for epoch in range(no_epochs):
    y_prediction = model(X)

    loss = criterion(y_prediction, y)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 10 == 1:
        print(loss.item())

predicted = model(X).detach().numpy()
plt.figure()
plt.plot(X_numpy, y_numpy, 'r')
plt.plot(X_numpy, predicted, 'b')
plt.show()

