import os
import tensorflow as tf
from tensorflow.keras import layers
from argparse import Namespace

from models.generator import *
from models.discriminator import *

class CycleGAN():
    def __init__(self, params):
        # Checkpoint paths
        checkpoint_paths = {}
        checkpoint_paths['G1'] = 'checkpoint/G1/'
        checkpoint_paths['G2'] = 'checkpoint/G2/'
        checkpoint_paths['D1'] = 'checkpoint/D1/'
        checkpoint_paths['D2'] = 'checkpoint/D2/'
        for path in checkpoint_paths.values():
            if not os.path.exists(path):
                os.makedirs(path)
        checkpoint_path = Namespace(**checkpoint_path)

        # G1: X -> Y
        # G2: Y -> X
        self.G1 = Generator(params.out_nc, params.nfg)
        self.G2 = Generator(params.out_nc, params.nfg)
        self.D1 = Discriminator(params.ndf, params.n_layers)
        self.D2 = Discriminator(params.ndf, params.n_layers)

        # discriminator loss function
        self.D_Loss = tf.keras.losses.mse

    def gan_loss(self, y, x, choice=1):
        # if choice==1:
        #     return tf.mean(tf.log(self.D1(y))) + tf.mean(tf.log(1 - self.D1(self.G1(x))))
        # return tf.mean(tf.log(self.D2(x))) + tf.mean(tf.log(1 - self.D2(self.G2(y))))
        if choice == 1:
            d1_fake = self.D1(self.G1(x))
            valid = tf.ones(d1_fake.shape, tf.float32)
            loss = self.D_Loss(d1_fake, valid)
        else:
            d2_fake = self.D2(self.G2(y))
            valid = tf.ones(d2_fake.shape, tf.float32)
            loss = self.D_Loss(d2_fake, valid)
        return loss

    def cycle_loss(self, y, x):
        return tf.mean(tf.norm(self.G2(self.G1(x)) - x, ord=1)) + \
               tf.mean(tf.norm(self.G1(self.G2(y)) - y, ord=1))

    def total_loss(self, y, x, lmbda):
        return 0.5 * (self.gan_loss(y, x, choice=1) + self.gan_loss(y, x, choice=2)) + \
               lmbda * self.cycle_loss(y, x)

    def discriminator_loss(self, y, x, choice=1):
        if choice == 1:
            d1_choice = self.D1(self.G1(x))
            d1_answer = tf.zeros(d1_choice.shape, tf.float32)
            loss_fake = self.D_Loss(d1_choice, d1_answer)
            d1_choice = self.D1(y)
            d1_answer = tf.ones(d1_choice.shape, tf.float32)
            loss_true = self.D_Loss(d1_choice, d1_answer)
        else:
            d2_choice = self.D2(self.G2(y))
            d2_answer = tf.zeros(d2_choice.shape, tf.float32)
            loss_fake = self.D_Loss(d2_choice, d2_answer)
            d2_choice = self.D2(x)
            d2_answer = tf.ones(d2_choice.shape, tf.float32)
            loss_true = self.D_Loss(d2_choice, d2_answer)
        return 0.5 * (loss_true + loss_fake)

    def save(self, epoch):
        self.G1.save_weights(os.path.join(checkpoint_paths.G1, f'cp_{epoch:04d}.ckpt'))
        self.G2.save_weights(os.path.join(checkpoint_paths.G2, f'cp_{epoch:04d}.ckpt'))
        self.D1.save_weights(os.path.join(checkpoint_paths.D1, f'cp_{epoch:04d}.ckpt'))
        self.D2.save_weights(os.path.join(checkpoint_paths.D2, f'cp_{epoch:04d}.ckpt'))

    def load(self, params):
        latestG1 = tf.train.latest_checkpoint(checkpoint_paths.G1)
        self.G1.load_weights(latestG1)
        latestG2 = tf.train.latest_checkpoint(checkpoint_paths.G2)
        self.G2.load_weights(latestG2)
        latestD1 = tf.train.latest_checkpoint(checkpoint_paths.D1)
        self.D1.load_weights(latestD1)
        latestD2 = tf.train.latest_checkpoint(checkpoint_paths.D2)
        self.D2.load_weights(latestD2)

    def __call__(self, x, y):
        return self.G1(x), self.G2(y)
