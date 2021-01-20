import torch
from torch import nn
from torch.nn import functional as F

class ARCNN(nn.Module):
    def __init__(self, weight):
        super().__init__()

        # PyTorch's Conv2D uses zero-padding while the matlab code uses replicate
        # So we need to use separate padding modules
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=7)
        self.conv22 = nn.Conv2d(32, 16, kernel_size=1)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=5)

        self.pad2 = nn.ReplicationPad2d(2)
        self.pad3 = nn.ReplicationPad2d(3)
        self.pad4 = nn.ReplicationPad2d(4)

        self.relu = nn.ReLU(inplace=True)

        # Load the weights from the weight dict
        self.conv1.weight.data = torch.from_numpy(
            weight['weights_conv1']
            .transpose(2, 0, 1)
            .reshape(64, 1, 9, 9)
            .transpose(0, 1, 3, 2)
        ).float()
        self.conv1.bias.data = torch.from_numpy(
            weight['biases_conv1']
            .reshape(64)
        ).float()

        self.conv2.weight.data = torch.from_numpy(
            weight['weights_conv2']
            .transpose(2, 0, 1)
            .reshape(32, 64, 7, 7)
            .transpose(0, 1, 3, 2)
        ).float()
        self.conv2.bias.data = torch.from_numpy(
            weight['biases_conv2']
            .reshape(32)
        ).float()

        self.conv22.weight.data = torch.from_numpy(
            weight['weights_conv22']
            .transpose(2, 0, 1)
            .reshape(16, 32, 1, 1)
            .transpose(0, 1, 3, 2)
        ).float()
        self.conv22.bias.data = torch.from_numpy(
            weight['biases_conv22']
            .reshape(16)
        ).float()

        self.conv3.weight.data = torch.from_numpy(
            weight['weights_conv3']
            .reshape(1, 16, 5, 5)
            .transpose(0, 1, 3, 2)
        ).float()
        self.conv3.bias.data = torch.from_numpy(
            weight['biases_conv3']
            .reshape(1)
        ).float()

    def forward(self, x):
        x = self.pad4(x)
        x = self.relu(self.conv1(x))

        x = self.pad3(x)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv22(x))

        x = self.pad2(x)
        x = self.conv3(x)

        return x
