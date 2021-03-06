from itertools import cycle
import numpy as np
import numpy.linalg as la
import time
import tensorflow as tf
from tensorflow.keras import layers
import argparse

from models.CycleGAN import *
from utils.params import *
from utils.data_loader import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("params_path", help="param file", type=str)
    args = parser.parse_args()
    print("Loading params...")
    param = Params(args.params_path)

    # load data
    print("Loading data...")
    datasetx, datasety = get_train_dataset(param)

    print("Creating model...")
    model = CycleGAN(param)
    optimizer = tf.optimizers.Adam(param.lr,beta_1=param.beta_1)

    for e in range(param.epochs):
        print(f'Starting epoch {e+1}')
        epoch_start_time = time.time()
        avgloss = 0.0
        datasetx = datasetx.shuffle(buffer_size=param.buff_size).batch(param.batch_size)
        datasety = datasety.shuffle(buffer_size=param.buff_size).batch(param.batch_size)
        for i, (data_x, data_y) in enumerate(zip(cycle(datasetx), datasety)):
            data_x = tf.reshape(data_x, [-1,param.image_size,param.image_size,param.in_nc])
            data_y = tf.reshape(data_y, [-1,param.image_size,param.image_size,param.in_nc])
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

            avgloss += loss

            if i%100 == 0:
                print(f'Training loss at epoch {e+1} step {i}: {float(avgloss / (i + 1))}')
        epoch_end_time = time.time()
        print(f'Epoch {e+1} training time: {epoch_end_time - epoch_start_time}')
        if (e + 1) % 10:
            model.save(e + 1)
