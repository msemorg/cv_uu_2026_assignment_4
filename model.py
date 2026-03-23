# lenet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary


# model1: returns val acc: 61.03% on cifar10
class LeNet5_Baseline(
    nn.Module
):  # baseline model, modified to work with RGB images instead of pure BW images
    def __init__(self, num_classes=10):
        super().__init__()
        # Layer 1: Convolute: patternmatching (3 in (RGB), 6 out (6 pattern-groups), 5x5 kernel ('viewable' window of the image in pixels. This window 'scans' over the entire image) -> Output: 28x28 pixel image because the kernel size is 5 and the stride (stepsize of window) is 1 so it ignores the edges where a full view does not fit
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        # Layer 2: Maxpooling: summarizing/shrinking image. 2x2 kernel, takes the maximum value of these 4 pixels. -> Output: 14x14 because only 1/2 of the pixels in height and  width are taken into account
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 3: Convolute (6 in (the patterngroups from layer 2), 16 out (16 pattern-groups), 5x5 kernel) -> Output: 10x10
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # Layer 4: MaxPool2d (2x2) -> Output: 5x5
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 5: Fully Connected (16*5*5 in, 120 out) because the output of the previous layer is 16*5*5 (16 pattern groups, 5x5 pixels) -> Output: 120 neurons
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # Layer 6: Fully Connected (120 in, 84 out) -> Output: 84 neurons
        self.fc2 = nn.Linear(120, 84)
        # Layer 7: Fully Connected (84 in, 10 out) -> Output: 10 neurons because we have 10 classes
        self.fc3 = nn.Linear(84, num_classes)

        # relu activation function to do non-linear transformations
        self.relu = nn.ReLU()

        # Kaiming Initialization for random starting numbers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # No softmax is used because we use CrossEntropyLoss instead
        x = self.fc3(x)
        return x


models = [LeNet5_Baseline]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model_class in models:
    model = model_class().to(device)
    print(f"*** Summary of {model_class.__name__} ***")
    summary(model, (3, 32, 32), device=str(device))