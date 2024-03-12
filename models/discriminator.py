import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__ (self):
        super(Discriminator, self).__init__()
        kernel_size = 5
        layer_filters = [32,64,128,256] # number of filters in each layer

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=layer_filters[0], kernel_size=kernel_size, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)  # Adding dropout with a dropout rate of 0.3
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=layer_filters[0], out_channels=layer_filters[1], kernel_size=kernel_size, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)  # Adding dropout with a dropout rate of 0.3
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=layer_filters[1], out_channels=layer_filters[2], kernel_size=kernel_size, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)  # Adding dropout with a dropout rate of 0.3
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=layer_filters[2], out_channels=layer_filters[3], kernel_size=kernel_size, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)  # Adding dropout with a dropout rate of 0.3
        )
        self.fc = nn.Linear(4 * 4 * layer_filters[3], 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = torch.flatten(x, 1) # flatten the tensor
        x = self.fc(x)
        x = torch.sigmoid(x) # sigmoid activation function to return value between 0 and 1
        return x