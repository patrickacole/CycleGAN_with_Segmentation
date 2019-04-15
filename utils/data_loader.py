import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf

def get_train_dataset(params, preprocess=True):
    """
    loads training data
    @param params   : Params object
    @return dataset : tf.data.Dataset object
    """
    x = load_dir(params.train_path_a,params.image_size,preprocess=preprocess)
    y = load_dir(params.train_path_b,params.image_size,preprocess=preprocess)
    return tf.data.Dataset.from_tensor_slices((x, y))

def get_test_dataset(params):
    """
    loads testing data
    @param params   : Params object
    @return dataset : tf.data.Dataset object
    """
    x = preprocess(load_dir(params.test_path_a,params.image_size))
    y = preprocess(load_dir(params.test_path_b,params.image_size))
    return tf.data.Dataset.from_tensor_slices((x, y))

def load_dir(path, size, preprocess=True):
    """
    loads images from a given path as a given size
    @param path : string for path to image directory
    @param size : int for size desired of images
    @return imgs: np.array of shape (n,size,size,3)
    """
    if not os.path.exists(path):
        print(f'Given path {path} does not exist...')
        sys.exit(1)
    imgs = []
    for file in os.listdir(path):
        if 'jpg' not in file:
            continue
        filename = os.path.join(file,path)
        img = Image.open(filename)
        img = img.resize((size,size),Image.BILINEAR)
        img = np.array(img)
        if preprocess:
            img = preprocess(img)
        imgs += [img]
    return np.array(imgs)

def preprocess(img):
    """
    transform image to be of range [-1,1]
    @param img  : numpy array of shape (size,size,3)
    @return img : numpy array of shape (size,size,3)
    """
    img = (img - np.mean(img,axis=-1)[np.newaxis,np.newaxis,:]) \
           / np.std(img,axis=-1)[np.newaxis,np.newaxis,:]
    return img

