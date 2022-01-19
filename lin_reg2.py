import torch as T
import numpy as np
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) Data processing
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

num_samples, num_features = X.shape
input_shape = num_features
output_shape = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1234)
# scale data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = T.from_numpy(X_train.astype(np.float32))
y_train = T.from_numpy(y_train.astype(np.float32))
X_test = T.from_numpy(X_test.astype(np.float32))
y_test = T.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1) Model

class LogisticRegression(nn.Module):
    def __init__(self, in_shape, out_shape):
        super(LogisticRegression, self).__init__()
        self.model = nn.Linear(in_shape, out_shape)

    def forward(self, x):
        activation = nn.Sigmoid()
        return activation(self.model(x))


model = LogisticRegression(input_shape, output_shape)
# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = T.optim.SGD(model.parameters(), lr=learning_rate)
# 3) backward
num_epochs = 1000

for epoch in range(num_epochs):
    y_predicted = model.forward(X_train)
    loss = criterion(y_predicted, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch+1) % 100 == 0:
        print(f'{loss.item():.4f}')

# Evaluation
with T.no_grad():
    y_test_prediction = model.forward(X_test)
    y_test_predicted_classes = y_test_prediction.round()
    acc = y_test_predicted_classes.eq(y_test).sum() / float(y_test.shape[0])
    print(f'Accuracy {acc:.4f}')


