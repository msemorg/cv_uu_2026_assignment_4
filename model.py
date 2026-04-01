import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


# Total: about  1,060,000 parameters
class NN_model(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # C1- in_channels=3 (RGB images), 16 kernels=out-channels(pattern groups), kernel_size=3 (3x3 pixel windowsize sliding over image), stride=1(stepsize), padding=1(add 1 pixel padding border)
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(16)
        # kernel_size=2x2, stride 2, padding 0
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # C2- 32 kernels=outchannels
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # C3- 64 kernels=outchannels
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # C4- 64 kernels=outchannels
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # FINAL CONV LAYER- 32 kernels=outchannels
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.bn5 = nn.BatchNorm2d(32)
        # flatten
        self.flatten = nn.Flatten()
        # dropout
        self.dropout = nn.Dropout(0.5)
        # fully connected layer: 512 neurons
        self.fc1 = nn.Linear(32 * 7 * 7, 512)
        # output layer: 343 neurons
        self.fc_out = nn.Linear(512, 343)
        self.relu = nn.ReLU()

        # Kaiming Initialization for random starting numbers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc_out(x)
        x = torch.sigmoid(x)
        return x.view(-1, 7, 7, 7)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NN_model().to(device)

summary(model, input_size=(1, 3, 112, 112))
