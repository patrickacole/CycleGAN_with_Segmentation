import numpy as np
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Defines a PatchGAN discriminator
    """

    def __init__(self, input_nc, ndf, n_layers):
        super(Discriminator, self).__init__()

        model = [nn.Conv2d(input_nc, ndf, 4, stride=2, padding=1),
                 nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        mult = ndf
        for i in range(1, n_layers):
            mult_prev = mult
            mult = min(2**i, 8) * ndf
            model += [nn.Conv2d(mult_prev, mult, kernel_size=4, stride=2, padding=1),
                      nn.InstanceNorm2d(mult),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        mult_prev = mult
        mult = min(2**n_layers, 8)
        model += [nn.Conv2d(mult_prev, mult, kernel_size=4, stride=1, padding=1),
                  nn.InstanceNorm2d(mult),
                  nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        model += [nn.Conv2d(mult, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    print("Testing discriminator...")
    input_nc = 1
    ndf = 4
    n_layers = 3

    device = torch.device(("cpu","cuda")[torch.cuda.is_available()])

    model = Discriminator(input_nc, ndf, n_layers).to(device)
    x = torch.zeros((1,3,256,256), dtype=torch.float32)
    y = model(x.to(device))
    print(y.shape)
