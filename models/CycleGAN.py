import tensorflow as tf
from tensorflow.keras import layers

from generator import *
from discriminator import *

class CycleGAN():
    def __init__(self, out_nc, nfg, ndf, n_layers):
        # G1: X -> Y
        # G2: Y -> X
        self.G1 = Generator(out_nc, nfg)
        self.G2 = Generator(out_nc, nfg)
        self.D1 = Discriminator(ndf, n_layers)
        self.D2 = Discriminator(ndf, n_layers)

        # discriminator loss function
        self.D_Loss = tf.keras.losses.mse

    def gan_loss(self,y, x, choice=1):
        if choice==1:
            return tf.mean(tf.log(self.D1(y))) + tf.mean(tf.log(1 - self.D1(self.G1(x))))
        return tf.mean(tf.log(self.D2(x))) + tf.mean(tf.log(1 - self.D2(self.G2(y))))

    def cycle_loss(self, y, x):
        return tf.mean(tf.norm(self.G2(self.G1(x)) - x, ord=1)) + \
               tf.mean(tf.norm(self.G1(self.G2(y)) - y, ord=1))

    def total_loss(self, y, x, lmbda):
        return self.gan_loss(y, x, choice=1) + self.gan_loss(y, x, choice=2) + \
               lmbda * self.cycle_loss(y, x)

    def discriminator_loss(self, y, x, choice=1):
        if choice == 1:
            d1_choice = self.D1(self.G1(x))
            d1_answer = tf.zeros(d1.shape, tf.float32)
            loss_fake = self.D_Loss(d1_choice, d1_answer)
            d1_choice = self.D1(y)
            d1_answer = tf.ones(d1.shape, tf.float32)
            loss_true = self.D_Loss(d1_choice, d1_answer)
        else:
            d2_choice = self.D2(self.G2(y))
            d2_answer = tf.zeros(d2.shape, tf.float32)
            loss_fake = self.D_Loss(d2_choice, d2_answer)
            d2_choice = self.D2(x)
            d2_answer = tf.ones(d2.shape, tf.float32)
            loss_true = self.D_Loss(d2_choice, d2_answer)
        return 0.5 * (loss_true + loss_fake)

    def save(self):
        # TODO
        pass

    def __call__(self, x, y):
        return self.G1(x), self.G2(y)