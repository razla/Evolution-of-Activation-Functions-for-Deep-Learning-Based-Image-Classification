import torch.nn as nn

class FullyConnected(nn.Module):
    def __init__(self, activation_functions, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(512, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, output_dim)
        self.af1 = activation_functions[0]
        self.af2 = activation_functions[1]
        self.af3 = activation_functions[2]
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)

    def forward(self, x):
        y = self.af1(self.bn1(self.linear1(x)))
        y = self.af2(self.bn2(self.linear2(y)))
        y = self.af3(self.bn3(self.linear3(y)))

        y = self.linear4(y)
        return y