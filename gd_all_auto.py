import torch as T
import torch.nn as nn


X = T.tensor([[1], [2], [3], [4]], dtype=T.float32)
Y = T.tensor([[2], [4], [6], [8]], dtype=T.float32)

n_samples, n_features = X.shape

X_test = T.tensor([5], dtype=T.float32)

model = nn.Linear(n_features, n_features)
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

lr = 0.01
n_iter = 500
loss = nn.MSELoss()
optimizer = T.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(n_iter):
    y_pred = model(X)
    l = loss(Y, y_pred)
    l.backward()
    [w, b] = model.parameters()

    # dw = w.grad
    # print(dw)

    optimizer.step()

    w.grad.zero_()

    if epoch % 5 == 0:
        print(f'epoch {epoch+1}: w = {w.item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')