import torch
import torch.nn as nn

class VanillaAE(nn.Module):
    def __init__(self):
        super().__init__()
        # self.input_size = 3 * 96 * 96
        self.input_size = 1 * 28 * 28

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, self.input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        input_size = x.size()
        x = x.view(input_size[:1] + (-1, ))
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(input_size)

        return x