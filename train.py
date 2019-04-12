import numpy as np
import numpy.linalg as la
import time
import tensorflow as tf
from tensorflow.keras import layers

from models.generator import *
from models.discriminator import *

def train(out_nc, ngf, ndf, n_layers):
    G1 = Generator(out_nc, nfg)
    G2 = Generator(out_nc, nfg)
    D1 = Discriminator(ndf, n_layers)
    D2 = Discriminator(ndf, n_layers)

class Loss():
    def __init__(out_nc, nfg, ndf, n_layers):
        self.G1 = Generator(out_nc, nfg)
        self.G2 = Generator(out_nc, nfg)
        self.D1 = Discriminator(ndf, n_layers)
        self.D2 = Discriminator(ndf, n_layers)

    def gan_loss(y, x, choice=1):
        if choice==1:
            return tf.mean(tf.log(self.D1(y))) + tf.mean(tf.log(1 - self.D1(self.G1(x))))
        return tf.mean(tf.log(self.D2(y))) + tf.mean(tf.log(1 - self.D2(self.G2(x))))

    def cycle_loss(y, x):
        return tf.mean(tf.norm(self.G2(self.G1(x)) - x, ord=1)) + \
               tf.mean(tf.norm(self.G1(self.G2(y)) - 1, ord=1))

    def loss(y, x, lmbda, choice=1):
        return gan_loss(y, x, choice=1) + lmbda * cycle_loss(y, x)

if __name__ == "__main__":
    # load data
    dataset = 

    out_nc = 
    ngf = 
    ndf = 
    n_layers = 
    # model = train(out_nc, ngf, ndf, n_layers)

    loss = Loss(out= , nfg= , ndf= , n_layers= )

    for e in epoch:
        epoch_start_time = time.time()

        for i, data in enumerate(dataset):
