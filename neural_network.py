import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, activation_functions, output_dim=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.af1 = activation_functions[0]
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.af2 = activation_functions[1]
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(1600, 1000)
        self.af3 = activation_functions[2]
        self.fc2 = nn.Linear(1000, output_dim)

    def forward(self, x):
        out = self.max_pool(self.af1(self.conv1(x)))
        out = self.max_pool(self.af2(self.conv2(out)))
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.af3(self.fc1(out))
        out = self.fc2(out)
        return out


class NeuralNet(nn.Module):
    def __init__(self, activation_functions, output_dim=10):
        super().__init__()
        self.linear1 = nn.Linear(784, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 32)
        self.linear4 = nn.Linear(32, 32)
        self.linear5 = nn.Linear(32, 32)
        self.linear6 = nn.Linear(32, 32)
        self.linear7 = nn.Linear(32, 32)
        self.classifier = nn.Linear(32, output_dim)
        self.afs = activation_functions

    def forward(self, x):
        out = self.afs[0](self.linear1(x.view(-1, 784)))
        out = self.afs[1](self.linear2(out))
        out = self.afs[1](self.linear3(out))
        out = self.afs[1](self.linear4(out))
        out = self.afs[1](self.linear5(out))
        out = self.afs[1](self.linear6(out))
        out = self.afs[2](self.linear7(out))
        out = self.classifier(out)
        return out