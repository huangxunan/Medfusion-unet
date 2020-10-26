
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import  merge, UpSampling2D, Dropout, Cropping2D, BatchNormalization
from keras.initializers import RandomNormal, VarianceScaling

import numpy as np



from keras.layers.core import Layer
from keras.engine import InputSpec
from keras.legacy import interfaces
from keras.utils import conv_utils
from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints


import tensorflow as tf
#from keras.layers import Conv2D
from keras.initializers import RandomNormal
from deform_conv import tf_batch_map_offsets
from os import path
from keras import Model, Input
from keras.callbacks import ModelCheckpoint, History
from keras.models import load_model
#from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import MaxPooling2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.optimizers import Adam
#from tumor import seg_data as data



#Adopted from https://github.com/pietz/unet-keras
#Added kernel initializers based on VarianceScaling


class _Conv(Layer):
    """Abstract nD convolution layer (private, used as implementation base).

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of outputs.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.

    # Arguments
        rank: An integer, the rank of the convolution,
            e.g. "2" for 2D convolution.
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of n integers, specifying the
            dimensions of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the strides of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, ..., channels)` while `"channels_first"` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    """

    def __init__(self, rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(_Conv, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank,
                                                      'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank,
                                                        'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        if self.rank == 1:
            outputs = K.conv1d(
                inputs,
                self.kernel,
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0])
        if self.rank == 2:
            outputs = K.conv2d(
                inputs,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.rank == 3:
            outputs = K.conv3d(
                inputs,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
        elif self.data_format == 'channels_first':
            space = input_shape[2:]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        if self.data_format == 'channels_last':
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        elif self.data_format == 'channels_first':
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(_Conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Conv2Dcopy(_Conv):
    @interfaces.legacy_conv2d_support
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super( Conv2Dcopy, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=False,
            kernel_initializer='zeros',
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
    def get_config(self):
        config = super(Conv2Dcopy, self).get_config()
        config.pop('rank')
        return config
    def call(self, x):
        # TODO offsets probably have no nonlinearity?
        x_shape = x.get_shape()
        offsets = super(Conv2Dcopy, self).call(x)

        offsets = self._to_bc_h_w_2(offsets, x_shape)
        x = self._to_bc_h_w(x, x_shape)
        x_offset = tf_batch_map_offsets(x, offsets)
        x_offset = self._to_b_h_w_c(x_offset, x_shape)
        return x_offset

    def compute_output_shape(self, input_shape):
        return input_shape

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, h, w, 2c) -> (b*c, h, w, 2)"""
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, (-1, int(x_shape[1]), int(x_shape[2]), 2))
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, h, w, c) -> (b*c, h, w)"""
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, (-1, int(x_shape[1]), int(x_shape[2])))
        return x

    @staticmethod
    def _to_b_h_w_c(x, x_shape):
        """(b*c, h, w) -> (b, h, w, c)"""
        x = tf.reshape(
            x, (-1, int(x_shape[3]), int(x_shape[1]), int(x_shape[2]))
        )
        x = tf.transpose(x, [0, 2, 3, 1])
        return x




def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., acti='relu',
		 dropout=0.0, batchnorm=False, maxpool=True, upconv=False, residual=False):
    init = VarianceScaling(scale=1.0 / 9.0)
    i = Input(shape=img_shape)
    n = Conv2D(8, 3, activation=acti, padding='same', kernel_initializer=init)(i)
    n = BatchNormalization()(n)
    #n = Conv2Dcopy(8 * 2, (3, 3), padding='same', kernel_initializer='zeros')(n)
    n = Conv2D(8, 3, activation=acti, padding='same', kernel_initializer=init)(n)
    n = BatchNormalization()(n)
    #n = Conv2Dcopy(8*2, (3, 3), padding='same', kernel_initializer='zeros')(n)
    co1 = concatenate([n, i], axis=3)

    m1 = MaxPooling2D()(co1)
    n = Conv2D(16, 3, activation=acti, padding='same', kernel_initializer=init)(m1)
    n = BatchNormalization()(n)
    n = Conv2D(16, 3, activation=acti, padding='same', kernel_initializer=init)(n)
    n = BatchNormalization()(n)
    #n = Conv2Dcopy(16 * 2, (3, 3), padding='same', kernel_initializer='zeros')(n)
    co2 = concatenate([n, m1], axis=3)

    m2 = MaxPooling2D()(co2)
    n = Conv2D(32, 3, activation=acti, padding='same', kernel_initializer=init)(m2)
    n = BatchNormalization()(n)
    n = Conv2D(32, 3, activation=acti, padding='same', kernel_initializer=init)(n)
    n = BatchNormalization()(n)
    #n = Conv2Dcopy(32 * 2, (3, 3), padding='same', kernel_initializer='zeros')(n)
    co3 = concatenate([n, m2], axis=3)

    m3 = MaxPooling2D()(co3)
    n = Conv2D(64, 3, activation=acti, padding='same', kernel_initializer=init)(m3)
    n = BatchNormalization()(n)
    n = Conv2D(64, 3, activation=acti, padding='same', kernel_initializer=init)(n)
    n = BatchNormalization()(n)
    #n = Conv2Dcopy(64 * 2, (3, 3), padding='same', kernel_initializer='zeros')(n)
    co4 = concatenate([n, m3], axis=3)

    m4 = MaxPooling2D()(co4)
    n = Conv2D(128, 3, activation=acti, padding='same', kernel_initializer=init)(m4)
    n = BatchNormalization()(n)
    n = Conv2D(128, 3, activation=acti, padding='same', kernel_initializer=init)(n)
    n = BatchNormalization()(n)
    #n = Conv2Dcopy(128 * 2, (3, 3), padding='same', kernel_initializer='zeros')(n)
    co5 = concatenate([n, m4], axis=3)

    m5 = MaxPooling2D()(co5)
    n = Conv2D(256, 3, activation=acti, padding='same', kernel_initializer=init)(m5)
    n = BatchNormalization()(n)
    n = Conv2D(256, 3, activation=acti, padding='same', kernel_initializer=init)(n)
    n = BatchNormalization()(n)
    co6 = concatenate([n, m5], axis=3)

    m6 = MaxPooling2D()(co6)
    n = Conv2D(512, 3, activation=acti, padding='same', kernel_initializer=init)(m6)
    n = BatchNormalization()(n)
    n = Conv2D(512, 3, activation=acti, padding='same', kernel_initializer=init)(n)
    n = BatchNormalization()(n)
    co7 = concatenate([n, m6], axis=3)

    m7 = MaxPooling2D()(co7)
    n = Conv2D(1024, 3, activation=acti, padding='same', kernel_initializer=init)(m7)
    n = BatchNormalization()(n)
    n = Conv2D(1024, 3, activation=acti, padding='same', kernel_initializer=init)(n)
    n = BatchNormalization()(n)
    co8 = concatenate([n, m7], axis=3)

    n = Conv2DTranspose(512, 3, strides=2, activation=acti, padding='same')(co8)
    co9 = concatenate([n, co7], axis=3)

    n = Conv2D(512, 3, activation=acti, padding='same', kernel_initializer=init)(co9)
    n = BatchNormalization()(n)
    n = Conv2D(512, 3, activation=acti, padding='same', kernel_initializer=init)(n)
    n = BatchNormalization()(n)
    co10 = concatenate([n, co9], axis=3)

    n = Conv2DTranspose(256, 3, strides=2, activation=acti, padding='same')(co10)
    co11 = concatenate([n, co6], axis=3)

    n = Conv2D(256, 3, activation=acti, padding='same', kernel_initializer=init)(co11)
    n = BatchNormalization()(n)
    n = Conv2D(256, 3, activation=acti, padding='same', kernel_initializer=init)(n)
    n = BatchNormalization()(n)
    co12 = concatenate([n, co11], axis=3)

    n = Conv2DTranspose(128, 3, strides=2, activation=acti, padding='same')(co12)
    co13 = concatenate([n, co5], axis=3)

    n = Conv2D(128, 3, activation=acti, padding='same', kernel_initializer=init)(co13)
    n = BatchNormalization()(n)
    n = Conv2D(128, 3, activation=acti, padding='same', kernel_initializer=init)(n)
    n = BatchNormalization()(n)
    co14 = concatenate([n, co13], axis=3)



    n = Conv2DTranspose(64, 3, strides=2, activation=acti, padding='same')(co14)
    co15 = concatenate([n, co4], axis=3)

    n = Conv2D(64, 3, activation=acti, padding='same', kernel_initializer=init)(co15)
    n = BatchNormalization()(n)
    n = Conv2D(64, 3, activation=acti, padding='same', kernel_initializer=init)(n)
    n = BatchNormalization()(n)
    co16 = concatenate([n, co15], axis=3)



    n = Conv2DTranspose(32, 3, strides=2, activation=acti, padding='same')(co16)
    co17 = concatenate([n, co3], axis=3)

    n = Conv2D(32, 3, activation=acti, padding='same', kernel_initializer=init)(co17)
    n = BatchNormalization()(n)
    n = Conv2D(32, 3, activation=acti, padding='same', kernel_initializer=init)(n)
    n = BatchNormalization()(n)
    co18 = concatenate([n, co17], axis=3)

    n = Conv2DTranspose(16, 3, strides=2, activation=acti, padding='same')(co18)
    co19 = concatenate([n, co2], axis=3)

    n = Conv2D(16, 3, activation=acti, padding='same', kernel_initializer=init)(co19)
    n = BatchNormalization()(n)
    n = Conv2D(16, 3, activation=acti, padding='same', kernel_initializer=init)(n)
    n = BatchNormalization()(n)
    co20 = concatenate([n, co19], axis=3)

    n = Conv2DTranspose(16, 3, strides=2, activation=acti, padding='same')(co20)
    co21 = concatenate([n, co1], axis=3)

    n = Conv2D(8, 3, activation=acti, padding='same', kernel_initializer=init)(co21)
    n = BatchNormalization()(n)
    n = Conv2D(8, 3, activation=acti, padding='same', kernel_initializer=init)(n)
    n = BatchNormalization()(n)
    co22 = concatenate([n, co21], axis=3)

    o = Conv2D(1, 1, activation='sigmoid')(co22)

    return Model(inputs=i, outputs=o)


'''  
def UNet1(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
		 dropout=0.0, bn=False, maxpool=True, upconv=False, residual=False):
	inputs = Input(shape=img_shape)
	conv1 = Conv2D(dim, (3, 3), activation='relu', padding='same')(inputs)
    n = BatchNormalization()(conv1) if bn else conv1
    conv2 = Conv2D(dim, 3, activation=acti, padding='same', kernel_initializer=init)(n)
    n = BatchNormalization()(n) if bn else n

'''


if  __name__=='__main__':

    model = UNet((256, 256, 1), start_ch=8, depth=1, batchnorm=True, dropout=0.5, upconv=True, maxpool=True,
                  residual=True)
    model.summary()
