import tensorflow as tf
import os

def get_train_dataset(params, pprocess=True):
    """
    loads training data
    @param params   : Params object
    @return dataset : tf.data.Dataset object
    """
    x = tf.data.Dataset.from_tensor_slices(load_dir(params.train_path_a))
    x_ds = x.map(lambda f: preprocess(f, params.image_size, pprocess=pprocess))
    y = tf.data.Dataset.from_tensor_slices(load_dir(params.train_path_b))
    y_ds = y.map(lambda f: preprocess(f, params.image_size, pprocess=pprocess))
    return x_ds, y_ds

def get_test_dataset(params, pprocess=True):
    """
    loads testing data
    @param params   : Params object
    @return dataset : tf.data.Dataset object
    """
    x = tf.data.Dataset.from_tensor_slices(load_dir(params.test_path_a))
    x_ds = x.map(lambda f: preprocess(f, params.image_size, pprocess=pprocess))
    y = tf.data.Dataset.from_tensor_slices(load_dir(params.test_path_b))
    y_ds = y.map(lambda f: preprocess(f, params.image_size, pprocess=pprocess))
    return (x, x_ds), (y, y_ds)

def load_dir(path):
    """
    builds list of images names from a given path
    @param path : string for path to image directory
    """
    if not os.path.exists(path):
        raise IOError(f'Given path {path} does not exist...')
    fnames = []
    for file in os.listdir(path):
        if 'jpg' not in file:
            continue
        fnames.append(os.path.join(path, file))
    return fnames

def preprocess(filename, size, pprocess=False):
    """
    transform image to be of range [-1,1] with correct size
    @param filename  : filename
    @return image : tensor of shape (size,size,3)
    """
    img_raw = tf.io.read_file(filename)
    #img_tens = tf.image.decode_image(img_raw)
    image = tf.image.decode_jpeg(img_raw, channels=3)
    image = tf.image.resize(image, [size, size])
    if pprocess:
        image /= 128.0  # normalize to [0,2] range
        image -= 1.0 # normalize to [-1,1] range
    return image

