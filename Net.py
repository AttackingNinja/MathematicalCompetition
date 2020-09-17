import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 5)

    def forward(self, x):
        x = self.fc1(x)
        return x
