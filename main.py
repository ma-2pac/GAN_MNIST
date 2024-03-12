import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch
import torch.nn as nn
import numpy as np

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


#defining the training loop
def train(generator, discriminator,gan, trainloader, epochs=50):

    #setup training parameters
    generator.train()
    discriminator.train()
    criterion = nn.BCELoss() # binary cross entropy loss
    lr = 2e-4 #learning rate
    decay = 6e-8 # weight decay
    factor = 0.5 # factor by which the learning rate will be reduced
    #optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=lr*factor, weight_decay=decay*factor) # RMSprop optimizer
    #optimizer_d = torch.optim.RMSprop(discriminator.parameters(), lr=lr*factor, weight_decay=decay*factor)
    
    # Setup Adam optimizers for both G and D
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))


    #training loop
    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(trainloader):

            #getting images
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, 1, requires_grad=False).to(device)
            fake_labels = torch.zeros(batch_size, 1, requires_grad=False).to(device)
            z = torch.randn(batch_size, 100).to(device)

            #training the discriminator
            optimizer_d.zero_grad() # zero the gradients
            generator.eval() # set the generator to evaluation mode
            with torch.no_grad():
                fake_images = generator(z) # generate fake images

            real_out = discriminator(real_images) # pass real images through the discriminator
            fake_out = discriminator(fake_images)
            loss_real = criterion(real_out, real_labels) # calculate loss for real images
            loss_fake = criterion(fake_out, fake_labels)
            loss_d = loss_real + loss_fake # total loss
            loss_d.backward() # backpropagation
            optimizer_d.step()

            #training the generator
            optimizer_g.zero_grad()
            generator.train()
            z = torch.randn(batch_size, 100, requires_grad=True).to(device)

            fake_out = gan(z) # pass the fake images through the GAN
            loss_g = criterion(fake_out, real_labels)
            loss_g.backward()
            optimizer_g.step()
            if i % 100 == 0:
                print(f'Epoch [{epoch}/{epochs}], Step [{i}/{len(trainloader)}], Loss D: {loss_d.item()}, Loss G: {loss_g.item()}')
        print('Learning Rate (Generator):', optimizer_g.param_groups[0]['lr'])
        print('Learning Rate (Discriminator):', optimizer_d.param_groups[0]['lr'])
           
        save_generated_img(generator,device,epoch=epoch)


#running the train loop
latent_dim = 100
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)
print(generator)
print(discriminator)
gan = GAN(generator, discriminator).to(device)
batch_size = 64
epochs = 50
train(generator, discriminator, gan, train_loader, epochs=epochs)