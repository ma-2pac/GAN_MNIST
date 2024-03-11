import torch
import torch.nn as nn

class GAN( nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, z):
        img = self.generator(z)
        validity = self.discriminator(img)
        return validity