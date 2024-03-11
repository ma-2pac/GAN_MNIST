import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    # nn.module is the base class of all NN modules

    def __init__(self, latent_dim):
        super(Generator, self).__init__() # super() returns a temporary object of the superclass that allows us to call that superclass’s methods
        kernel_size  = 5 # kernel size
        self.blocks = nn.ModuleList() # ModuleList is a list of modules. It can be indexed like a regular Python list and will be visible by all Module methods.
        self.fc = nn.Linear(latent_dim, 7 * 8 * 128) # fully connected layer
        self.block1 = nn.Sequential(
            nn.BatchNorm2d(128), # batch normalization
            nn.ReLU(True), # activation function
            nn.ConvTranspose2d(128, 128, kernel_size, stride=2, padding=2, output_padding=1), # transpose convolutional layer
        )
        self.block2 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size, stride=2, padding=2, output_padding=1),
        )
        self.block3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size, stride=1, padding=2),
        )
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size, stride=1, padding=2),
            nn.Sigmoid(),
        )

    def forward(self, z): # forward pass
        x = self.fc(z) # fully connected layer
        x = x.view(-1, 128, 7, 7) # reshape
        x = self.block1(x) # pass through block1
        x = self.block2(x)
        x = self.block3(x)
        img = self.block4(x) # output
        return img

