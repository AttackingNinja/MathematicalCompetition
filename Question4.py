import torch.nn as nn
import torch
from torch import optim

from Net import Net

model = Net()
net_loss = nn.CrossEntropyLoss()
data = torch.tensor([39.26, 17.38, 9.56, 11.09])
output = model(data)
target = torch.tensor([0, 0, 1, 0, 0], dtype=torch.long)
loss = net_loss(model(data), target)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(model.parameters())
