import numpy as np
import numpy.linalg as la
import time
import tensorflow as tf
from tensorflow.keras import layers
import sys

from models.CycleGAN import *
from utils.params import *

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 train.py <params_path>")
        exit(1)
    param = Params(sys.argv[1])

    # load data
    x_train = ...
    y_train = ...
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(buffer_size=param.buffer_size).batch(param.batch_size)

    model = CycleGAN(out=param.out_nc, ngf=param.ngf, \
                     ndf=param.ndf, n_layers=param.n_layers)
    optimizers = tf.optimizers.Adam(param.lr)

    for e in range(param.epoch):
        print(f'Starting epoch {e+1}')
        epoch_start_time = time.time()

        for i, (data_x, data_y) in enumerate(dataset):
            with tf.GradientTape() as tape:
                loss = model.total_loss(data_y, data_x, param.lmbda)
                loss_d1 = model.discriminator_loss(data_y, data_x, choice=1)
                loss_d2 = model.discriminator_loss(data_y, data_x, choice=2)

            gradG1 = tape.gradient(loss, (model.G1).trainable_variables)
            gradG2 = tape.gradient(loss, (model.G2).trainable_variables)
            gradD1 = tape.gradient(loss_d1, (model.D1).trainable_variables)
            gradD2 = tape.gradient(loss_d2, (model.D2).trainable_variables)

            optimizer.apply_gradients(zip(gradG1, (model.G1).trainable_variables))
            optimizer.apply_gradients(zip(gradG2, (model.G2).trainable_variables))
            optimizer.apply_gradients(zip(gradD1, (model.D1).trainable_variables))
            optimizer.apply_gradients(zip(gradD2, (model.D2).trainable_variables))

            if i%200 == 0:
                print(f'Training loss at step {i}: {float(loss)}')
