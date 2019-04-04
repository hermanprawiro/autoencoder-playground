import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision

from models import vae

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    learning_rate = 1e-3
    weight_decay = 1e-5
    batch_size = 128
    start_epoch = 0
    max_epoch = 5

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    # trainset = torchvision.datasets.STL10('./data', split='train', transform=train_transforms)
    # testset = torchvision.datasets.STL10('./data', split='test', transform=train_transforms)
    trainset = torchvision.datasets.MNIST('./data', transform=train_transforms)
    testset = torchvision.datasets.MNIST('./data', train=False, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    model = vae.VAE_MNIST().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    criterion = vae.VAE_Loss().to(device)

    for epoch in range(start_epoch, max_epoch):
        train(model, trainloader, criterion, optimizer, epoch)
    # visualize(model, testloader, criterion)
    visualize2(model)

def train(model, loader, criterion, optimizer, epoch):
    model.train()
    total_iter = len(loader)

    for i, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        inputs = inputs.view(-1, 784)

        outputs, mu, logvar = model(inputs)
        loss = criterion(outputs, inputs, mu, logvar)

        if i % 20 == 0:
            print('Epoch {epoch}\t[{iter}/{total_iter}]:\tLoss {loss:.5f}'.format(epoch=epoch + 1, iter=i + 1, total_iter=total_iter, loss=loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def visualize2(model):
    model.eval()

    # inputs = torch.randn(64, 20).to(device)
    inputs = []
    z_range = torch.linspace(-2, 2, 16)
    for i, yi in enumerate(z_range):
        for j, xi in enumerate(z_range):
            inputs.append(torch.tensor([[xi, yi]]))
    inputs = torch.cat(inputs).to(device)
    outputs = model.decode(inputs).data.cpu()

    plt.imshow(np.transpose(torchvision.utils.make_grid(outputs.view(-1, 1, 28, 28), nrow=16), (1, 2, 0)))
    plt.show()

def visualize(model, loader, criterion):
    model.eval()

    num_shown = 8
    output_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage()
    ])

    inputs, targets = next(iter(loader))
    inputs = inputs.to(device)
    inputs = inputs.view(-1, 784)

    outputs, mu, logvar = model(inputs)
    loss = criterion(outputs[:num_shown], inputs[:num_shown], mu, logvar)
    print('Visualize Loss {loss:.5f}'.format(loss=loss.item()))

    inputs = inputs.data[:num_shown].cpu()
    outputs = outputs.data[:num_shown].cpu()

    for i in range(num_shown):
        plt.subplot(4, 4, i + 1)
        plt.imshow(output_transforms(inputs[i].view(28, 28)))

    for i in range(num_shown):
        plt.subplot(4, 4, i + 9)
        plt.imshow(output_transforms(outputs[i].view(28, 28)))
    plt.show()

if __name__ == "__main__":
    main()