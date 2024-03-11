import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#import models
from models.generator import Generator
from models.discriminator import Discriminator
from models.gan import GAN

#import dataloader
from dataloader import train_loader


def save_generated_img(generator,device,epoch=1):

    #generate random noise vectors
    noise= torch.randn(16,100).to(device)

    #generate images with generator
    with torch.no_grad(): # no need to calculate gradients
        generated_img = generator(noise).detach().cpu()

    #create a grid of 4x4 images
    grid  =vutils.make_grid(generated_img, nrow=4, padding=2, normalize=True)


    #save the grid of images
    filename = f'imgs/gan_epoch-{epoch}.png'
    vutils.save_image(grid, filename)