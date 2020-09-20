import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=4)
        self.fc1 = nn.Linear(4, 5)
        # self.fc2 = nn.Linear(20, 10)
        # self.fc3 = nn.Linear(10, 5)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.view([x.shape[0], x.shape[2]])
        x = self.fc1(x)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x
