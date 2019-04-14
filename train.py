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
    dataset = tf.data.Dataset.from_tensor_slices()
    dataset = dataset.shuffle(buffer_size=param.buff_size).batch(param.batch_size)

    out_nc = param.out_nc
    ngf = param.ngf
    ndf = param.ndf
    n_layers = param.n_layers

    model = CycleGAN(out=out_nc , ngf=ngf , ndf=ndf , n_layers=n_layers)
    optimizers = tf.optimizers.Adam(param.lr)

    for e in range(param.epoch):
        print(f'Starting epoch {e+1}')
        epoch_start_time = time.time()

        for i, (data_x, data_y) in enumerate(dataset):
            with tf.GradientTape() as tape:
                loss = model.total_loss(data_y, data_x, param.lmbda)
            gradient1 = tape.gradient(loss, (model.G1).trainable_variables)
            gradient2 = tape.gradient(loss, (model.G2).trainable_variables)

            optimizer.apply_gradients(zip(gradient1, (model.G1).trainable_variables))
            optimizer.apply_gradients(zip(gradient2, (model.G2).trainable_variables))

            if i%200 == 0:
                print(f'Training loss at step {i}: {float(loss)}')
