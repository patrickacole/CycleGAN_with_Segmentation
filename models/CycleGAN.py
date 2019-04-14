import numpy as np
import numpy.linalg as la
import tensorflow as tf
from tensorflow.keras import layers

from generator import *
from discriminator import *

class CycleGAN():
    def __init__(out_nc, nfg, ndf, n_layers):
        # G1: X -> Y
        # G2: Y -> X
        self.G1 = Generator(out_nc, nfg)
        self.G2 = Generator(out_nc, nfg)
        self.D1 = Discriminator(ndf, n_layers)
        self.D2 = Discriminator(ndf, n_layers)

    def gan_loss(y, x, choice=1):
        if choice==1:
            return tf.mean(tf.log(self.D1(y))) + tf.mean(tf.log(1 - self.D1(self.G1(x))))
        return tf.mean(tf.log(self.D2(x))) + tf.mean(tf.log(1 - self.D2(self.G2(y))))

    def cycle_loss(y, x):
        return tf.mean(tf.norm(self.G2(self.G1(x)) - x, ord=1)) + \
               tf.mean(tf.norm(self.G1(self.G2(y)) - y, ord=1))

    def total_loss(y, x, lmbda):
        return gan_loss(y, x, choice=1) + gan_loss(y, x, choice=2) + lmbda * cycle_loss(y, x)