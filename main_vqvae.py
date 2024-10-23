# Import required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the VQ-VAE model components
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, x):
        # Flatten input except batch dimension
        flat_x = x.reshape(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_x**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_x, self.embedding.weight.t()))
        
        # Get nearest embedding indices
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Convert indices to one-hot
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Quantize the inputs
        quantized = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized.reshape(x.shape)
        
        # Compute loss
        commitment_loss = F.mse_loss(quantized.detach(), x)
        embedding_loss = F.mse_loss(quantized, x.detach())
        vq_loss = commitment_loss + 0.25 * embedding_loss
        
        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        
        return quantized, vq_loss

class Encoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dims=[32, 64], embedding_dim=64):
        super().__init__()
        
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[-1], embedding_dim, kernel_size=1),
                nn.BatchNorm2d(embedding_dim),
                nn.LeakyReLU()
            )
        )
        
        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, embedding_dim=64, hidden_dims=[64, 32], out_channels=1):
        super().__init__()
        
        modules = []
        in_channels = embedding_dim
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
            
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            )
        )
        
        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.decoder(x)

class VQVAE(nn.Module):
    def __init__(self, num_embeddings=32, embedding_dim=64, img_channels=1):
        super().__init__()
        self.encoder = Encoder(embedding_dim=embedding_dim, in_channels=img_channels) # 4x downsampling
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder(embedding_dim=embedding_dim, out_channels=img_channels)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss = self.vector_quantizer(z)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss

# Data loading and preprocessing
def get_data_loaders(dataset_name, batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset_name == 'mnist':
        dataset_class = torchvision.datasets.MNIST
    elif dataset_name == 'cifar10':
        dataset_class = torchvision.datasets.CIFAR10
    else:
        raise ValueError('Unknown dataset')
    
    train_dataset = dataset_class(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )
    
    test_dataset = dataset_class(
        root='./data',
        train=False,
        transform=transform,
        download=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def denormalize(tensor):
    transform = transforms.Compose([
        transforms.Normalize(-1, 2),
        transforms.ToPILImage()
    ])
    return transform(tensor)


# Training function
def train_epoch(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, vq_loss = model(data)
        recon_loss = F.mse_loss(recon_batch, data)
        loss = recon_loss + vq_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    avg_loss = total_loss / len(train_loader)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    return avg_loss

# Visualization function
def visualize_results(model, test_loader, epoch, dataset_name, num_images=8):
    model.eval()
    with torch.no_grad():
        data = next(iter(test_loader))[0][:num_images].to(device)
        recon, _ = model(data)
        
        # Plot original and reconstructed images
        fig, axes = plt.subplots(2, num_images, figsize=(2*num_images, 4))
        
        for i in range(num_images):
            # Original images
            axes[0, i].imshow(denormalize(data[i].cpu()), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original')
            
            # Reconstructed images
            axes[1, i].imshow(denormalize(recon[i].cpu()), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed')
        
        plt.tight_layout()
        plt.savefig(f'vqvae_{dataset_name}_epoch_{epoch}.png')
        # plt.show()

def visualize_embeddings(model, epoch, dataset_name):
    model.eval()
    with torch.no_grad():
        # Get the embeddings
        dummy_z = model.vector_quantizer.embedding.weight.data[:, :, None, None]
        dummy_z = dummy_z.expand(-1, -1, 7, 7)
        recon = model.decoder(dummy_z)
        
        images = torchvision.utils.make_grid(recon, nrow=4, normalize=True)
        plt.subplots(figsize=(10, 10))
        plt.imshow(images.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.savefig(f'vqvae_{dataset_name}_embeddings_{epoch}.png')
        # plt.show()

# Main training loop
def main():
    # Hyperparameters
    batch_size = 128
    num_epochs = 10
    learning_rate = 1e-3

    num_embeddings = 16 # codebook size
    embedding_dim = 4

    dataset_name = 'mnist'
    if dataset_name == 'mnist':
        img_channels = 1
    else:
        img_channels = 3
    
    # Initialize model and optimizer
    model = VQVAE(num_embeddings, embedding_dim, img_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(dataset_name, batch_size)
    
    # Training loop
    losses = []
    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, epoch)
        losses.append(loss)
        
        visualize_embeddings(model, epoch, dataset_name)
        if epoch % 2 == 0:
            visualize_results(model, test_loader, epoch, dataset_name)
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'vqvae_{dataset_name}_loss.png')
    # plt.show()
    
    # Save the model
    torch.save(model.state_dict(), f'vqvae_{dataset_name}.pth')

if __name__ == "__main__":
    main()