import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_MNIST(nn.Module):
    def __init__(self, input_size=784, hidden_size=400, latent_size=2):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)

        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        mu = self.fc21(x)
        logvar = self.fc22(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, x):
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x = self.decode(z)
        return x, mu, logvar

class VAE_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_criterion = nn.BCELoss(reduction='sum')
    
    def forward(self, inputs, targets, mu, logvar):
        BCE = self.bce_criterion(inputs, targets)

        # KL Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - torch.pow(mu, 2) - logvar.exp())

        return BCE + KLD