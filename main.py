import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision

from models import linear, conv2d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    learning_rate = 1e-3
    weight_decay = 1e-5
    batch_size = 128
    start_epoch = 0
    max_epoch = 2

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    # trainset = torchvision.datasets.STL10('./data', transform=train_transforms)
    trainset = torchvision.datasets.MNIST('./data', transform=train_transforms)
    testset = torchvision.datasets.MNIST('./data', train=False, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    # model = linear.VanillaAE().to(device)
    model = conv2d.Conv2DAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()

    for epoch in range(start_epoch, max_epoch):
        train(model, trainloader, criterion, optimizer, epoch)
    visualize(model, testloader, criterion)

def train(model, loader, criterion, optimizer, epoch):
    model.train()
    total_iter = len(loader)

    for i, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        if i % 20 == 0:
            print('Epoch {epoch}\t[{iter}/{total_iter}]:\tLoss {loss:.5f}'.format(epoch=epoch + 1, iter=i + 1, total_iter=total_iter, loss=loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def visualize(model, loader, criterion):
    model.eval()

    num_shown = 8
    output_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage()
    ])

    inputs, targets = next(iter(loader))
    inputs = inputs.to(device)

    outputs = model(inputs)
    loss = criterion(outputs[:num_shown], inputs[:num_shown])
    print('Visualize Loss {loss:.5f}'.format(loss=loss.item()))

    inputs = inputs.data[:num_shown].cpu()
    outputs = outputs.data[:num_shown].cpu()

    for i in range(num_shown):
        plt.subplot(4, 4, i + 1)
        plt.imshow(output_transforms(inputs[i]))

    for i in range(num_shown):
        plt.subplot(4, 4, i + 9)
        plt.imshow(output_transforms(outputs[i]))
    plt.show()

if __name__ == "__main__":
    main()