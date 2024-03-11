import torch
import torchvision
from torchvision import transforms

#define transformation to MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#load the dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)

#create dataloaders for training
batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
