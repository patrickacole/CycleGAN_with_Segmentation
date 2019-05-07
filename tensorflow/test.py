import os
import time
import argparse
import numpy as np
import numpy.linalg as la
import tensorflow as tf
from tensorflow.keras import layers

from models.CycleGAN import *
from utils.params import *
from utils.data_loader import *

def save_outputs(outputs, filenames_in, filenames_out, out_directory, pprocess=True):
    #raise NotImplementedError("This function needs to be verified before use")
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    for i in range(outputs.shape[0]):
        image = outputs[i]
        f_in, ext = os.path.splitext(os.path.basename(filenames_in[i]))
        f_out, ext = os.path.splitext(os.path.basename(filenames_out[i]))
        if pprocess:
            image += 1.0
            image *= 128.0
        img_raw = tf.image.encode_jpeg(image, format='rgb')
        fname = os.path.join(out_directory, f"{f_in}_{f_out}{ext}")
        tf.io.write_file(fname, img_raw)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("params_path", help="param file", type=str)
    args = parser.parse_args()
    print("Loading params...")
    param = Params(args.params_path)

    # load data
    print("Loading data...")
    x, y = get_test_dataset(param)
    filenamesx = x[0]
    datasetx = x[1]
    filenamesy = y[0]
    datasety = y[1]
    datasetx = datasetx.batch(param.batch_size)
    datasety = datasety.batch(param.batch_size)

    print("Creating model...")
    model = CycleGAN(param)
    model.load(param)

    print("Beginning to test...")
    for fx, data_x in zip(filenamesx,datasetx):
	print(fx.eval())
	print(data_x)
        for fy, data_y in zip(filenamesy,datasety):
            G_x_out, G_y_out = model(data_x, data_y)
            # Save G_x_out as fx->fy.jpg
            save_outputs(G_x_out, fx, fy, param.out_directory)
            # Save G_y_out as fy->fx.jpg
            save_outputs(G_y_out, fy, fx, param.out_directory)
