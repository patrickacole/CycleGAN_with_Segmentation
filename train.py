import numpy as np
import numpy.linalg as la
import time
import tensorflow as tf
from tensorflow.keras import layers
import sys
import os

from models.CycleGAN import *
from utils.params import *
from utils.data_loader import *

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 train.py <params_path>")
        exit(1)
    print("Loading params...")
    param = Params(sys.argv[1])

    # load data
    print("Loading data...")
    datasetx, datasety = get_train_dataset(param)
    datasetx = datasetx.shuffle(buffer_size=max(param.buff_size,tf.size(datasetx)[0])).batch(param.batch_size)
    datasety = datasety.shuffle(buffer_size=max(param.buff_size,tf.size(datasety)[0])).batch(param.batch_size)

    print("Creating model...")
    model = CycleGAN(params)
    optimizers = tf.optimizers.Adam(param.lr,beta_1=param.beta_1)

    for e in range(param.epochs):
        print(f'Starting epoch {e+1}')
        epoch_start_time = time.time()

        for i, data_x in enumerate(datasetx):
            for j, data_y in enumerate(datasety):
                with tf.GradientTape(persistent=True) as tape:
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
                print(f'Training loss at epoch {e+1} step {i}: {float(loss)}')

        if (e + 1) % 10:
            model.save(e + 1)
