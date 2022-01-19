import torch as T

X = T.tensor([1, 2, 3, 4], dtype=T.float32)
Y = T.tensor([2, 4, 6, 8], dtype=T.float32)

w = T.tensor(0.0, dtype=T.float32, requires_grad=True)


def forward(input):
    output = input * w

    return output


def loss(y, y_pred):
    return ((y-y_pred)**2).mean()


def gradient(x, y, y_pred):
    return T.dot(2*x, y_pred-y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

lr = 0.01
n_iter = 80

for epoch in range(n_iter):
    y_pred = forward(X)
    l = loss(Y, y_pred)
    l.backward()

    dw = w.grad
    # print(dw)

    with torch.no_grad():
        w -= lr * dw

    w.grad.zero_()
    # print(w.grad)

    if epoch % 5 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')