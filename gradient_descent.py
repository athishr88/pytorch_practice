import numpy as np

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0


def forward(input):
    output = input * w

    return output


def loss(y, y_pred):
    return ((y-y_pred)**2).mean()


def gradient(x, y, y_pred):
    return np.dot(2*x, y_pred-y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

lr = 0.01
n_iter = 10

for epoch in range(n_iter):
    y_pred = forward(X)
    l = loss(Y, y_pred)
    dw = gradient(X, Y, y_pred)

    w -= lr * dw

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')