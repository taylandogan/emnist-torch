import torch.nn as nn
import torch.nn.functional as F


class EMNISTModel(nn.Module):

    def __init__(self):
        super(EMNISTModel, self).__init__()
        # Input data size: (batch_size, channels, height, width)
        # (16, 1, 28, 28)

        # inputs: (16, 1, 28, 28)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # outputs: (16, 10, 24, 24)

        # inputs: (16, 10, 24, 24)
        self.mp1 = nn.MaxPool2d(kernel_size=2)
        # outputs: (16, 10, 12, 12)

        # inputs: (16, 10, 12, 12)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # outputs: (16, 20, 8, 8)

        # inputs: (16, 1280)
        self.fc1 = nn.Linear(1280, 256)
        # outputs: (16, 256)

        # inputs: (16, 256)
        self.fc2 = nn.Linear(256, 62)
        # inputs: (16, 62)


    def forward(self, x):
        # print("Shape1: ", x.shape)
        x = self.conv1(x)
        x = F.relu(self.mp1(x))
        # print("Shape2: ", x.shape)
        x = F.relu(self.conv2(x))
        # print("Shape3: ", x.shape)
        x = x.view(-1, 1280)
        # print("Jape: ", x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)