import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class ResidualLayer(nn.Module):
    """
    Residual layer containing two convolutions
    """
    def __init__(self, ngf, **kwargs):
        super(ResidualLayer, self).__init__(**kwargs)
        self.model = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(ngf, ngf, 3),
                nn.ReLU(),
                nn.ReflectionPad2d(1),
                nn.Conv2d(ngf, ngf, 3)
                )

    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    """
    Generator architecture from CycleGAN paper

    Contains:
        A convolution with kernel size 7
        Two downsampling layers (using Conv2D w/ strides=2)
        Six residual layers
        Two upsampling layers (using Conv2DTranspose w/ strides=2)
        A convolution with kernel size 7
    """
    def __init__(self, output_nc, ngf, residual_layers = 3):
        super(Generator, self).__init__()
        d = []
        d += [nn.ReflectionPad2d(3)]
        d += [nn.Conv2d(3, ngf, kernel_size=7)]
        d += [nn.InstanceNorm2d(ngf)]
        d += [nn.ReLU()]
        n_downsampling = 2
        #add downsampling layers
        for i in range(n_downsampling):
            mult = 2 ** i
            d += [nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3, stride=2, padding=1)]
            d += [nn.InstanceNorm2d(ngf*mult*2)]
            d += [nn.ReLU()]
        #residual layers
        for _ in range(residual_layers):
            d += [ResidualLayer(ngf*(2**n_downsampling))]
        # add upsampling layers
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            d += [nn.ConvTranspose2d(ngf*mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1)]
            d += [nn.InstanceNorm2d(int(ngf*mult/2)), nn.ReLU()]

        d += [
                nn.ReflectionPad2d(3), 
                nn.Conv2d(ngf, output_nc, kernel_size=7),
                nn.Tanh()
                ]
        self.model = nn.Sequential(*d)

    def forward(self, inputs):
        "Assumes inputs are (N, C, H, W)"
        return self.model(inputs)

        

if __name__=="__main__":
    model = Generator(3, 1)
    t = torch.tensor(np.reshape(np.arange(3*256*256), [1, 3, 256, 256]), dtype=torch.float)
    #test case, a single 256 x 256 image with 1 channel
    print(model(t).shape)
