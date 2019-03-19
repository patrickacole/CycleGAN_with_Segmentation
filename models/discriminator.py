import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from generator import InstanceNorm2D

class Discriminator(tf.keras.Model):
    '''
    Discriminator based on PatchGAN discriminator used in CycleGAN paper 
    '''
    def __init__(self, ndf, n_layers=3):
        super(Discriminator, self).__init__(name='discriminator')
        self.model = tf.keras.Sequential()
        self.model.add(layers.Conv2D(filters=ndf, kernel_size=5, strides=2, padding='valid', use_bias=True, data_format='channels_last'))
        self.model.add(layers.LeakyReLU(alpha=0.2))
        for i in range(1, n_layers):
            mult = min(2 ** (i + 1), 8)
            self.model.add(layers.Conv2D(filters=ndf*mult, kernel_size=3, strides=2, padding='valid', use_bias=True, data_format='channels_last'))
            self.model.add(InstanceNorm2D())
            self.model.add(layers.LeakyReLU(alpha=0.2))
        self.model.add(layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='valid', use_bias=True, data_format='channels_last'))             
 
    def call(self, inputs):
        return self.model(inputs)

if __name__=="__main__":
    model = Discriminator(2)
    t = tf.constant(np.reshape(np.arange(256*256), [1, 256, 256, 1]), dtype=tf.float32)
    #test case, a single 256 x 256 image with 1 channel
    print(model(t).shape)
