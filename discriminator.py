import torch
import torch.nn as nn

class Discriminator3D(nn.Module):
    def __init__(self):
        super(Discriminator3D, self).__init__()

        # Conv Block 1: 4x4x4, 64 filters
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.batch_norm1 = nn.BatchNorm3d(64)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)

        # Conv Block 2: 4x4x4, 128 filters
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.batch_norm2 = nn.BatchNorm3d(128)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)

        # Conv Block 3: 4x4x4, 256 filters
        self.conv3 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.batch_norm3 = nn.BatchNorm3d(256)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)

        # Flatten Layer
        self.flatten = nn.Flatten()

        # Fully connected layer
        self.fc = nn.Linear(256 * 8 * 8 * 8, 1)  # Adjust based on 64x64x64 input

        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass through the Conv Block 1
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu1(x)

        # Pass through the Conv Block 2
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.leaky_relu2(x)

        # Pass through the Conv Block 3
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.leaky_relu3(x)

        # Flatten
        x = self.flatten(x)

        # Fully connected layer
        x = self.fc(x)

        # Sigmoid activation
        output = self.sigmoid(x)

        return output

