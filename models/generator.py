import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class ReflectionPad2D(layers.Layer):
    """
    Reflection Padding Layer

    Pads an image with format channels_last using reflection
    """
    def __init__(self, padding, **kwargs):
        """
        @param padding: a 4-tuple, or int padding
        """
        super(ReflectionPad2D, self).__init__(name='reflectionPad2D')
        if isinstance(padding, int):
            self.pad = [[padding] * 2] * 2
        elif isinstance(padding, tuple) and len(padding)==4:
            self.pad = [[padding[0], padding[1]], [padding[2], padding[3]]]
        else:
            raise ValueError("padding expected to be 4-tuple or int, but has type " + type(padding))
        super(ReflectionPad2D, self).__init__(**kwargs)

    def build(self, input_shape):
        rank = input_shape.rank
        if rank == 2:
            padding = self.pad
        else:
            padding = ([[0, 0]]*(rank-3)) + self.pad + [[0, 0]]
        self.padding = tf.constant(padding, tf.int32)
        super(ReflectionPad2D, self).build(input_shape)

    def call(self, inputs):
        x = tf.pad(inputs, self.padding, "REFLECT")
        return x

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-3] = shape[-3] + sum(self.padding[0])
        shape[-2] = shape[-2] + sum(self.padding[1])
        return tf.TensorShape(shape)


class InstanceNorm2D(layers.Layer):
    """
    Normalize the outputs of an image with format channels_last
    
    Normalization done according to https://arxiv.org/pdf/1607.08022.pdf
    """
    def __init__(self, eps=1e-5, momentum=0.1, **kwargs):
        super(InstanceNorm2D, self).__init__(**kwargs)
        self.eps = tf.constant(eps)
        self.momentum = tf.constant(momentum)

    def build(self, input_shape):
        super(InstanceNorm2D, self).build(input_shape)

    def call(self, inputs):
        mean = tf.expand_dims(tf.expand_dims(tf.reduce_mean(inputs, axis=[-3, -2]), 1), 1)
        covariates = tf.expand_dims(tf.expand_dims(tf.reduce_mean(tf.square(inputs - mean), axis=[-3, -2]), 1), 1)
        x = (1-self.momentum) * (inputs - mean)/tf.sqrt(covariates+self.eps) + self.momentum*inputs
        return x

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)


class ResidualLayer(layers.Layer):
    """
    Residual layer containing two convolutions
    """
    def __init__(self, ngf, **kwargs):
        super(ResidualLayer, self).__init__(**kwargs)
        self.ngf = ngf

    def build(self, input_shape):
        self.pad1 = ReflectionPad2D(1)
        self.conv1 = layers.Conv2D(self.ngf, kernel_size=3, strides=1, padding='valid', use_bias=True)
        self.pad2 = ReflectionPad2D(1)
        self.conv2 = layers.Conv2D(self.ngf, kernel_size=3, strides=1, padding='valid', use_bias=True)
        for weight in self.conv1.trainable_weights:
            self.add_weight(weight)
        for weight in self.conv2.trainable_weights:
            self.add_weight(weight)
        super(ResidualLayer, self).build(input_shape)

    def call(self, inputs):
        return self.conv2(self.pad2(tf.nn.relu(self.conv1(self.pad1(inputs))))) + inputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)


class Generator(tf.keras.Model):
    """
    Generator architecture from CycleGAN paper

    Contains:
        A convolution with kernel size 7
        Two downsampling layers (using Conv2D w/ strides=2)
        Six residual layers
        Two upsampling layers (using Conv2DTranspose w/ strides=2)
        A convolution with kernel size 7
    """
    def __init__(self, output_nc, ngf):
        super(Generator, self).__init__(name='generator')
        self.output_nc = output_nc
        self.model = tf.keras.Sequential()
        self.model.add(ReflectionPad2D(3))
        self.model.add(layers.Conv2D(filters=ngf, kernel_size=7, padding='valid', use_bias=True, data_format='channels_last'))
        self.model.add(InstanceNorm2D())
        self.model.add(layers.ReLU())
        n_downsampling = 2
        #add downsampling layers
        for i in range(n_downsampling):
            mult = 2 ** i
            self.model.add(layers.Conv2D(filters=ngf*mult*2, kernel_size=3, strides=2, padding='valid', use_bias=True, data_format='channels_last'))
            self.model.add(InstanceNorm2D())
            self.model.add(layers.ReLU())
        #residual layers
        for _ in range(6):
            self.model.add(ResidualLayer(ngf))
        # add upsampling layers
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            self.model.add(layers.Conv2DTranspose(int(ngf * mult / 2),
                kernel_size=3, strides=2, padding='valid',
                use_bias=True, data_format='channels_last',
                output_padding=1 if i==n_downsampling-1 else None))
            self.model.add(InstanceNorm2D())
            self.model.add(layers.ReLU())

        self.model.add(ReflectionPad2D(3))
        self.model.add(layers.Conv2D(filters=output_nc, kernel_size=7, padding='valid', use_bias=True, data_format='channels_last'))
        self.model.add(layers.Activation('tanh'))

    def call(self, inputs):
        return self.model(inputs)

    def compute_output_shape(self, input_shape):
        output_shape = tf.TensorShape(input_shape).as_list()
        output_shape[1] = self.output_nc
        return tf.TensorShape(output_shape)

        

if __name__=="__main__":
    model = Generator(2, 1)
    t = tf.constant(np.reshape(np.arange(256*256), [1, 256, 256, 1]), dtype=tf.float32)
    #test case, a single 256 x 256 image with 1 channel
    print(model(t).shape)
