import os
import itertools
import torch
import torch.nn as nn
from torch.autograd import Variable
from argparse import Namespace

# from generator import *
# from discriminator import *
from models.generator import *
from models.discriminator import *

class CycleGAN():
    def __init__(self, params, device):
        # G_AB: A -> B
        # G_BA: B -> A
        self.G_AB = Generator(params.out_nc, params.ngf).to(device)
        self.G_BA = Generator(params.out_nc, params.ngf).to(device)
        self.D_A = Discriminator(params.in_nc, params.ndf, params.n_layers).to(device)
        self.D_B = Discriminator(params.in_nc, params.ndf, params.n_layers).to(device)

        # discriminator loss function
        self.D_Loss = nn.MSELoss().to(device)
        self.L1_loss = nn.L1Loss().to(device)

        # set device to be used later
        self.device = device

        # optimizers
        self.optimizer_D_A = torch.optim.Adam(self.D_A.parameters(), lr = params.lr, betas=(params.beta_1, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.D_B.parameters(), lr = params.lr, betas=(params.beta_1, 0.999))
        self.optimizer_G  = torch.optim.Adam(itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), 
                                             lr=params.lr, betas=(params.beta_1, 0.999))

        # self.optimizer_G_AB = torch.optim.Adam(self.G_AB.parameters(), lr = params.lr, betas=(params.beta_1, 0.999))
        # self.optimizer_G_BA = torch.optim.Adam(self.G_BA.parameters(), lr = params.lr, betas=(params.beta_1, 0.999))

    # def gan_loss(self, realA, realB, choice='AB'):
    #     if choice == 'AB':
    #         db_fake = self.D_B(self.G_AB(realA))
    #         valid = torch.ones(db_fake.shape).to(self.device)
    #         loss = self.D_Loss(db_fake, valid)
    #     else:
    #         da_fake = self.D_A(self.G_BA(realB))
    #         valid = torch.ones(da_fake.shape).to(self.device)
    #         loss = self.D_Loss(da_fake, valid)
    #     return loss #tf.reduce_mean(loss)

    # def cycle_loss(self, realA, realB):
    #     return (self.L1_loss(self.G_BA(self.G_AB(realA)), realA) + self.L1_loss(self.G_AB(self.G_BA(realB)), realB)) / 2

    # def total_loss(self, realA, realB, lmbda):
    #     return (self.gan_loss(realA, realB, choice='AB') + self.gan_loss(realA, realB, choice='BA')) / 2 + \
    #            lmbda * self.cycle_loss(realA, realB)

    # This stuff doesnt calculate extra stuff :-)
    def gan_loss(self, fakeA, fakeB, choice='AB'):
        if choice == 'AB':
            db_fake = self.D_B(fakeB)
            valid = torch.ones(db_fake.shape).to(self.device)
            loss = self.D_Loss(db_fake, valid)
        else:
            da_fake = self.D_A(fakeA)
            valid = torch.ones(da_fake.shape).to(self.device)
            loss = self.D_Loss(da_fake, valid)
        return loss #tf.reduce_mean(loss)

    def cycle_loss(self, realA, realB, fakeA, fakeB):
        return (self.L1_loss(self.G_BA(fakeB), realA) + self.L1_loss(self.G_AB(fakeA), realB)) / 2

    def total_loss(self, realA, realB, lmbda):
        fakeB = self.G_AB(realA)
        fakeA = self.G_BA(realB)
        return (self.gan_loss(fakeA, fakeB, choice='AB') + self.gan_loss(fakeA, fakeB, choice='BA')) / 2 + \
               lmbda * self.cycle_loss(realA, realB, fakeA, fakeB)

    def discriminator_loss(self, realA, realB, choice='A'):
        if choice == 'A':
            da_choice = self.D_A(self.G_BA(realB))
            da_answer = torch.zeros(da_choice.shape).to(self.device)
            loss_fake = self.D_Loss(da_choice, da_answer)
            da_choice = self.D_A(realA)
            da_answer = torch.ones(da_choice.shape).to(self.device)
            loss_true = self.D_Loss(da_choice, da_answer)
        else:
            db_choice = self.D_B(self.G_AB(realA))
            db_answer = torch.zeros(db_choice.shape).to(self.device)
            loss_fake = self.D_Loss(db_choice, db_answer)
            db_choice = self.D_B(realB)
            db_answer = torch.ones(db_choice.shape).to(self.device)
            loss_true = self.D_Loss(db_choice, db_answer)
        return (loss_true + loss_fake) / 2

    def optimize_parameters(self, realA, realB, params):
        # Optimize Generators
        self.optimizer_G.zero_grad()
        self.g_loss = self.total_loss(realA, realB, params.lmbda)
        self.g_loss.backward()
        self.optimizer_G.step()

        # Optimize Discriminator A
        self.optimizer_D_A.zero_grad()
        self.d_a_loss = self.discriminator_loss(realA, realB, choice='A')
        self.d_a_loss.backward()
        self.optimizer_D_A.step()

        # Optimize Discriminator B
        self.optimizer_D_B.zero_grad()
        self.d_b_loss = self.discriminator_loss(realA, realB, choice='B')
        self.d_b_loss.backward()
        self.optimizer_D_B.step()

    def save(self, epoch, params):
        if not os.path.exists(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
                
        state = {'epoch':epoch + 1,'state_dict':self.G_AB.state_dict()}
        filepath = os.path.join(params.checkpoint_dir, 'G_AB.last.pth.tar')
        torch.save(state, filepath)

        state = {'epoch':epoch + 1,'state_dict':self.G_BA.state_dict()}
        filepath = os.path.join(params.checkpoint_dir, 'G_BA.last.pth.tar')
        torch.save(state, filepath)

        state = {'epoch':epoch + 1,'optim_dict':self.optimizer_G.state_dict()}
        filepath = os.path.join(params.checkpoint_dir, 'optimG.last.pth.tar')
        torch.save(state, filepath)

        state = {'epoch':epoch + 1,'state_dict':self.D_A.state_dict(),'optim_dict':self.optimizer_D_A.state_dict()}
        filepath = os.path.join(params.checkpoint_dir, 'D_A.last.pth.tar')
        torch.save(state, filepath)

        state = {'epoch':epoch + 1,'state_dict':self.D_B.state_dict(),'optim_dict':self.optimizer_D_B.state_dict()}
        filepath = os.path.join(params.checkpoint_dir, 'D_B.last.pth.tar')
        torch.save(state, filepath)

    def load(self, params):
        filepath = os.path.join(params.checkpoint_dir, 'G_AB.last.pth.tar')
        if not os.path.exists(filepath):
            raise IOError(f'File {filepath} does not exist')
        latestG_AB = torch.load(filepath)
        self.G_AB.load_state_dict(latestG_AB['state_dict'])
        # if params.optimizer:
        #     self.optimizer_G_AB.load_state_dict(latestG_AB['optim_dict'])

        filepath = os.path.join(params.checkpoint_dir, 'G_BA.last.pth.tar')
        if not os.path.exists(filepath):
            raise IOError(f'File {filepath} does not exist')
        latestG_BA = torch.load(filepath)
        self.G_BA.load_state_dict(latestG_BA['state_dict'])
        # if params.optimizer:
        #     self.optimizer_G_BA.load_state_dict(latestG_BA['optim_dict'])

        filepath = os.path.join(params.checkpoint_dir, 'optimG.last.pth.tar')
        if not os.path.exists(filepath):
            raise IOError(f'File {filepath} does not exist')
        latestOptimG = torch.load(filepath)
        if params.optimizer:
            self.optimizer_G.load_state_dict(latestOptimG['optim_dict'])

        filepath = os.path.join(params.checkpoint_dir, 'D_A.last.pth.tar')
        if not os.path.exists(filepath):
            raise IOError(f'File {filepath} does not exist')
        latestD_A = torch.load(filepath)
        self.D_A.load_state_dict(latestD_A['state_dict'])
        if params.optimizer:
            self.optimizer_D_A.load_state_dict(latestD_A['optim_dict'])

        filepath = os.path.join(params.checkpoint_dir, 'D_B.last.pth.tar')
        if not os.path.exists(filepath):
            raise IOError(f'File {filepath} does not exist')
        latestD_B = torch.load(filepath)
        self.D_B.load_state_dict(latestD_B['state_dict'])
        if params.optimizer:
            self.optimizer_D_B.load_state_dict(latestD_B['optim_dict'])

        return latestG_AB['epoch']

    def train(self):
        self.G_AB.train()
        self.G_BA.train()
        self.D_A.train()
        self.D_B.train()

    def eval(self):
        self.G_AB.eval()
        self.G_BA.eval()
        self.D_A.eval()
        self.D_B.eval()

    def __call__(self, x, y):
        return self.G_AB(x), self.G_BA(y)

if __name__ == "__main__":
    from argparse import Namespace
    args = {"checkpoint_dir":"checkpoint/",
            "in_nc":3,
            "out_nc":3,
            "ngf":16,
            "ndf":16,
            "n_layers":1,
            "lmbda":10,
            "lr":1e-4,
            "beta_1":0.5}
    args = Namespace(**args)

    device = 'cpu'
    t1 = torch.zeros(4, 3, 128, 128).to(device)
    t2 = torch.zeros(4, 3, 128, 128).to(device)

    c = CycleGAN(args, device)
    c.optimize_parameters(t1, t2, args)
    c.save(1, args)
    print(c(t1,t2)[0].shape)
