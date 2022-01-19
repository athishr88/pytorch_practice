import torch
import torch as t

x = t.tensor(1.0)
y = t.tensor(2.0)
w = t.tensor(1.0, requires_grad=True)

y_hat = x * w
s = y_hat - y
loss = s**2

loss.backward()
print(w.grad)