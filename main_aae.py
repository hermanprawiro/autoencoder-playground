import time
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision

from utils.metrics import AverageMeter
from utils.network import save_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AAE_Encoder(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, latent_size=2):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x), inplace=True)
        x = nn.functional.relu(self.fc2(x), inplace=True)
        x = self.fc3(x)
        return x

class AAE_Decoder(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, latent_size=2):
        super().__init__()

        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x), inplace=True)
        x = nn.functional.relu(self.fc2(x), inplace=True)
        x = torch.sigmoid(self.fc3(x))
        return x

class AAE_Discriminator(nn.Module):
    def __init__(self, latent_size=2, hidden_size=256):
        super().__init__()

        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x), inplace=True)
        x = nn.functional.relu(self.fc2(x), inplace=True)
        x = torch.sigmoid(self.fc3(x))
        return x

class AAE_DiscriminatorImage(nn.Module):
    def __init__(self, input_size=784, hidden_size=256):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x), inplace=True)
        x = nn.functional.relu(self.fc2(x), inplace=True)
        x = torch.sigmoid(self.fc3(x))
        return x


def generate_visualization_latent():
    inputs = []
    z_range = torch.linspace(-3, 3, 16)
    for i, yi in enumerate(z_range):
        for j, xi in enumerate(z_range):
            inputs.append(torch.tensor([[xi, yi]]))
    inputs = torch.cat(inputs)
    return inputs

def main():
    learning_rate = 1e-3
    weight_decay = 0
    batch_size = 64
    max_epoch = 5
    start_epoch = 0

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    trainset = torchvision.datasets.MNIST('./data', transform=train_transforms)
    testset = torchvision.datasets.MNIST('./data', train=False, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    encoder = AAE_Encoder().to(device)
    decoder = AAE_Decoder().to(device)
    discriminator = AAE_Discriminator().to(device)
    discriminator_image = AAE_DiscriminatorImage().to(device)

    opt_enc = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    opt_dec = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    opt_discimg = torch.optim.Adam(discriminator_image.parameters(), lr=learning_rate)

    criterion = nn.BCELoss()

    # checkpoint_prefix = 'aae_mnist_basic'

    # checkpoint_path = os.path.join('checkpoints', '{}.tar'.format(checkpoint_prefix))
    # if os.path.exists(checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    #     model.load_state_dict(checkpoint['state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     print('Checkpoint loaded, last epoch = {}'.format(checkpoint['epoch'] + 1))
    #     start_epoch = checkpoint['epoch'] + 1

    model = (encoder, decoder, discriminator, discriminator_image)
    optimizer = (opt_enc, opt_dec, opt_disc, opt_discimg)
    for epoch in range(start_epoch, max_epoch):
        train(trainloader, model, optimizer, criterion, None, epoch)
        # save_model(model, optimizer, epoch, checkpoint_prefix)

def train(loader, model, optimizer, criterion, writer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    R_losses = AverageMeter()
    G_losses = AverageMeter()
    D_losses = AverageMeter()
    GI_losses = AverageMeter()
    DI_losses = AverageMeter()

    encoder, decoder, discriminator, discriminator_image = model
    opt_enc, opt_dec, opt_disc, opt_discimg = optimizer
    encoder.train()
    decoder.train()
    discriminator.train()

    fixed_noise = generate_visualization_latent().to(device)

    real_label = 1
    fake_label = 0

    total_iter = len(loader)
    end = time.time()
    for i, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs.size(0)
        # Reconstruction
        opt_enc.zero_grad()
        opt_dec.zero_grad()
        
        inputs = inputs.view(batch_size, -1)
        outputs = decoder(encoder(inputs))
        lossR = criterion(outputs, inputs)
        
        lossR.backward()
        opt_dec.step()
        opt_enc.step()

        # Discriminator
        opt_disc.zero_grad()
        
        fake_latent = encoder(inputs)

        # Real
        real_latent = torch.empty_like(fake_latent).normal_()
        label = torch.full((batch_size,), real_label, device=device)
        outputs = discriminator(real_latent).view(-1)
        lossD_real = criterion(outputs, label)
        lossD_real.backward()

        # Fake
        label.fill_(fake_label)
        outputs = discriminator(fake_latent.detach()).view(-1)
        lossD_fake = criterion(outputs, label)
        lossD_fake.backward()
        lossD = lossD_real + lossD_fake
        opt_disc.step()

        # Generator (Encoder)
        opt_enc.zero_grad()

        label.fill_(real_label)
        outputs = discriminator(fake_latent).view(-1)
        lossG = criterion(outputs, label)
        lossG.backward()
        opt_enc.step()

        # Discriminator Image
        opt_discimg.zero_grad()
        fake_latent = encoder(inputs)
        # Real
        outputs = discriminator_image(inputs).view(-1)
        lossDI_real = criterion(outputs, label)
        lossDI_real.backward()

        # Fake
        label.fill_(fake_label)
        outputs = discriminator_image(decoder(fake_latent).detach()).view(-1)
        lossDI_fake = criterion(outputs, label)
        lossDI_fake.backward()
        lossDI = lossDI_real + lossDI_fake
        opt_discimg.step()

        # Generator (Decoder)
        opt_dec.zero_grad()

        label.fill_(real_label)
        outputs = discriminator_image(decoder(fake_latent.detach())).view(-1)
        lossGI = criterion(outputs, label)
        lossGI.backward()
        opt_dec.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # measure accuracy and record loss
        R_losses.update(lossR.item(), batch_size)
        D_losses.update(lossD.item(), batch_size)
        G_losses.update(lossG.item(), batch_size)
        DI_losses.update(lossDI.item(), batch_size)
        GI_losses.update(lossGI.item(), batch_size)

        # global_step = (epoch * total_iter) + i + 1
        # writer.add_scalar('train/loss', losses.val, global_step)

        if i % 10 == 0:
            print('Epoch {0} [{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss_R {lossR.val:.4f} ({lossR.avg:.4f})\t'
                'Loss_D {lossD.val:.4f} ({lossD.avg:.4f})\t'
                'Loss_G {lossG.val:.4f} ({lossG.avg:.4f})\t'
                'Loss_DI {lossDI.val:.4f} ({lossDI.avg:.4f})\t'
                'Loss_GI {lossGI.val:.4f} ({lossGI.avg:.4f})'.format(
                epoch + 1, i + 1, total_iter, 
                batch_time=batch_time, data_time=data_time, lossR=R_losses, lossD=D_losses, lossG=G_losses, lossDI=DI_losses, lossGI=GI_losses)
            )
            with torch.no_grad():
                fake = decoder(fixed_noise).detach().cpu()
                torchvision.utils.save_image(fake.view(-1, 1, 28, 28), 'results/aae_mnist/e{:02d}_i{:05d}.png'.format(epoch, i), nrow=16,padding=2, normalize=True)

if __name__ == "__main__":
    main()