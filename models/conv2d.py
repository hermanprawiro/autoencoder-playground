import torch
import torch.nn as nn

class Conv2DAE(nn.Module):
    def __init__(self):
        super().__init__()
        # self.input_size = 3 * 96 * 96
        self.input_size = 1 * 28 * 28

        # MNIST
        # Encoder
        # (n, 1, 28, 28)
        # (n, 16, 14, 14)
        # (n, 32, 7, 7)
        # (n, 64, 4, 4)
        # Decoder
        # (n, 32, 7, 7)
        # (n, 16, 14, 14)
        # (n, 1, 28, 28)
        self.encoder = nn.Sequential(
            ConvBlock(1, 16, kernel_size=3, stride=2, padding=1),
            ConvBlock(16, 32, kernel_size=3, stride=2, padding=1),
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=1)
        )

        self.decoder = nn.Sequential(
            DeConvBlock(64, 32, kernel_size=3, stride=2, padding=1),
            DeConvBlock(32, 16, kernel_size=4, stride=2, padding=1),
            DeConvBlock(16, 1, kernel_size=4, stride=2, padding=1, activation='sigmoid'),
        )

    def forward(self, x):
        input_size = x.size()
        x = self.encoder(x)
        x = self.decoder(x)

        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation='relu'):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)

        return x

class DeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation='relu'):
        super().__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)

        return x