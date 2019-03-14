import torch
import torch.nn as nn

class Conv2DAE(nn.Module):
    def __init__(self):
        super().__init__()
        # MNIST
        # Encoder (n, 1, 28, 28) [k=3, s=2, p=1] > (n, 16, 14, 14) [k=3, s=2, p=1] > (n, 32, 7, 7) [k=3, s=2, p=1]
        # Latent (n, 64, 4, 4)
        # Decoder (n, 32, 7, 7) [k=3, s=2, p=1] > (n, 16, 14, 14) [k=4, s=2, p=1] > (n, 1, 28, 28) [k=4, s=2, p=1]

        # STL-10
        # Encoder (n, 3, 96, 96) [k=3, s=2, p=1] > (n, 16, 48, 48) [k=3, s=2, p=1] > (n, 32, 24, 24) [k=3, s=2, p=1]
        # Latent (n, 64, 12, 12)
        # Decoder (n, 32, 24, 24) [k=3, s=2, p=1] > (n, 16, 48, 48) [k=3, s=2, p=0] > (n, 3, 96, 96) [k=3, s=2, p=1]
        # self.encoder = nn.Sequential(
        #     ConvBlock(1, 16, kernel_size=3, stride=2, padding=1),
        #     ConvBlock(16, 32, kernel_size=3, stride=2, padding=1),
        #     ConvBlock(32, 64, kernel_size=3, stride=2, padding=1)
        # )

        # self.decoder = nn.Sequential(
        #     ConvBlock(64, 32, kernel_size=3, stride=2, padding=1, deconv=True),
        #     ConvBlock(32, 16, kernel_size=4, stride=2, padding=1, deconv=True),
        #     ConvBlock(16, 1, kernel_size=4, stride=2, padding=1, deconv=True, activation='sigmoid'),
        # )
        self.encoder = nn.Sequential(
            ConvBlock(3, 32, kernel_size=4, stride=2, padding=1, activation='leaky'),
            ConvBlock(32, 64, kernel_size=4, stride=2, padding=1, activation='leaky'),
            ConvBlock(64, 128, kernel_size=4, stride=2, padding=1, activation='leaky')
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBlock(128, 64, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2),
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2),
            ConvBlock(32, 3, kernel_size=3, stride=1, padding=1, activation='sigmoid'),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation='relu', deconv=False):
        super().__init__()

        if deconv:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
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