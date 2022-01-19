import torch as T
import torch.nn as nn

loss = nn.CrossEntropyLoss()

Y = T.tensor([0])

y_good = T.tensor([[2.0, 0.2, 0.4]])
y_bad = T.tensor([[0.2, 0.5, 0.9]])

l1 = loss(y_bad, Y)
l2 = loss(y_good, Y)

print(l1, l2)

i, prediction1 = T.max(y_good, 1)
k, prediction2 = T.max(y_bad, 1)

print(prediction1)
print(prediction2)
