import os
import itertools
import torch
import torch.nn as nn
from torch.autograd import Variable
from argparse import Namespace

from generator import *
from discriminator import *
# from models.generator import *
# from models.discriminator import *

class CycleGAN():
    def __init__(self, params):
        # G1: X -> Y
        # G2: Y -> X
        self.G1 = Generator(params.out_nc, params.ngf)
        self.G2 = Generator(params.out_nc, params.ngf)
        self.D1 = Discriminator(params.in_nc, params.ndf, params.n_layers)
        self.D2 = Discriminator(params.in_nc, params.ndf, params.n_layers)

        # discriminator loss function
        self.D_Loss = nn.MSELoss
        self.L1_loss = nn.L1Loss

        # optimizers
        self.optimizer_D1 = torch.optim.Adam(self.D1.parameters(), lr = params.lr, betas=(params.beta_1, 0.999))
        self.optimizer_D2 = torch.optim.Adam(self.D2.parameters(), lr = params.lr, betas=(params.beta_1, 0.999))
        self.optimizer_G  = torch.optim.Adam(itertools.chain(self.G1.parameters(), self.G2.parameters()), 
                                             lr=params.lr, betas=(params.beta_1, 0.999))
        self.optimizer_G1 = torch.optim.Adam(self.G1.parameters(), lr = params.lr, betas=(params.beta_1, 0.999))
        self.optimizer_G2 = torch.optim.Adam(self.G2.parameters(), lr = params.lr, betas=(params.beta_1, 0.999))

    def gan_loss(self, y, x, choice=1):
        if choice == 1:
            print(self.G1(x))
            d1_fake = self.D1(self.G1(x))
            valid = torch.ones(d1_fake.shape)
            loss = self.D_Loss(d1_fake, valid)
        else:
            d2_fake = self.D2(self.G2(y))
            valid = torch.ones(d2_fake.shape)
            loss = self.D_Loss(d2_fake, valid)
        return loss #tf.reduce_mean(loss)

    def cycle_loss(self, y, x):
        return (self.L1_loss(self.G2(self.G1(x)), x) + self.L1_loss(self.G1(self.G2(y)), y))/2

    def total_loss(self, y, x, lmbda):
        return (self.gan_loss(y, x, choice=1) + self.gan_loss(y, x, choice=2))/2 + \
               lmbda * self.cycle_loss(y, x)

    def discriminator_loss(self, y, x, choice=1):
        if choice == 1:
            d1_choice = self.D1(self.G1(x))
            d1_answer = torch.zeros(d1_choice.shape)
            loss_fake = self.D_Loss(d1_choice, d1_answer)
            d1_choice = self.D1(y)
            d1_answer = torch.ones(d1_choice.shape)
            loss_true = self.D_Loss(d1_choice, d1_answer)
        else:
            d2_choice = self.D2(self.G2(y))
            d2_answer = torch.zeros(d2_choice.shape)
            loss_fake = self.D_Loss(d2_choice, d2_answer)
            d2_choice = self.D2(x)
            d2_answer = torch.ones(d2_choice.shape)
            loss_true = self.D_Loss(d2_choice, d2_answer)
        return (loss_true + loss_fake)/2

    def optimize_parameters(self, realA, realB, params):
        self.optimizer_G.zero_grad()
        self.g_loss = self.total_loss(realB, realA, params.lmbda)
        self.g_loss.backward()
        self.optimizer_G.step()

        self.optimizer_D1.zero_grad()
        self.d1_loss = self.discriminator_loss(realB, realA)
        self.d1_loss.backward()
        self.optimizer_D1.step()

        self.optimizer_D2.zero_grad()
        self.d2_loss = self.discriminator_loss(realB, realA, choice=2)
        self.d2_loss.backward()
        self.optimizer_D2.step()

    def save(self, epoch, params):
        if not os.path.exists(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
                
        state = {'epoch':epoch + 1,'state_dict':self.G1.state_dict()}
        filepath = os.path.join(params.checkpoint_dir, 'G1.last.pth.tar')
        torch.save(state, filepath)

        state = {'epoch':epoch + 1,'state_dict':self.G2.state_dict()}
        filepath = os.path.join(params.checkpoint_dir, 'G2.last.pth.tar')
        torch.save(state, filepath)

        state = {'epoch':epoch + 1,'optim_dict':self.optimizer_G.state_dict()}
        filepath = os.path.join(params.checkpoint_dir, 'optimG.last.pth.tar')
        torch.save(state, filepath)

        state = {'epoch':epoch + 1,'state_dict':self.D1.state_dict(),'optim_dict':self.optimizer_D1.state_dict()}
        filepath = os.path.join(params.checkpoint_dir, 'D1.last.pth.tar')
        torch.save(state, filepath)

        state = {'epoch':epoch + 1,'state_dict':self.D2.state_dict(),'optim_dict':self.optimizer_D2.state_dict()}
        filepath = os.path.join(params.checkpoint_dir, 'D2.last.pth.tar')
        torch.save(state, filepath)

    def load(self, params):
        filepath = os.path.join(params.checkpoint_dir, 'G1.last.pth.tar')
        if not os.path.exists(filepath):
            raise IOError(f'File {filepath} does not exist')
        latestG1 = torch.load(filepath)
        self.G1.load_state_dict(latestG1['state_dict'])
        # if params.optimizer:
        #     self.optimizer_G1.load_state_dict(latestG1['optim_dict'])

        filepath = os.path.join(params.checkpoint_dir, 'G2.last.pth.tar')
        if not os.path.exists(filepath):
            raise IOError(f'File {filepath} does not exist')
        latestG2 = torch.load(filepath)
        self.G2.load_state_dict(latestG2['state_dict'])
        # if params.optimizer:
        #     self.optimizer_G2.load_state_dict(latestG2['optim_dict'])

        filepath = os.path.join(params.checkpoint_dir, 'optimG.last.pth.tar')
        if not os.path.exists(filepath):
            raise IOError(f'File {filepath} does not exist')
        latestOptimG = torch.load(filepath)
        if params.optimizer:
            self.optimizer_g.load_state_dict(latestOptimG['optim_dict'])

        filepath = os.path.join(params.checkpoint_dir, 'D1.last.pth.tar')
        if not os.path.exists(filepath):
            raise IOError(f'File {filepath} does not exist')
        latestD1 = torch.load(filepath)
        self.D1.load_state_dict(latestD1['state_dict'])
        if params.optimizer:
            self.optimizer_D1.load_state_dict(latestD1['optim_dict'])

        filepath = os.path.join(params.checkpoint_dir, 'D2.last.pth.tar')
        if not os.path.exists(filepath):
            raise IOError(f'File {filepath} does not exist')
        latestD2 = torch.load(filepath)
        self.D2.load_state_dict(latestD2['state_dict'])
        if params.optimizer:
            self.optimizer_D2.load_state_dict(latestD2['optim_dict'])

        return latestG1['epoch']

    def train(self):
        self.G1.train()
        self.G2.train()
        self.D1.train()
        self.D2.train()

    def eval(self):
        self.G1.eval()
        self.G2.eval()
        self.D1.eval()
        self.D2.eval()

    def __call__(self, x, y):
        return self.G1(x), self.G2(y)

# ------------------------------------------------------------------------------------------
# TODO Remove after testing
import json

class Params():
    """
    Disclaimer. Taken from 
    https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py

    Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

if __name__ == "__main__":
    t1 = torch.zeros(4, 1, 256, 256)
    t2 = torch.zeros(4, 1, 256, 256)

    params = Params("params/training.json")
    c = CycleGAN(params)
    c.optimize_parameters(t1, t2, params)
    c.save(1, params)