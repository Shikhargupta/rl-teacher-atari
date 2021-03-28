from math import ceil
import sys

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow as tf2

class FullyConnectedMLP(object):
    """Vanilla two hidden layer multi-layer perceptron"""

    def __init__(self, obs_shape, act_shape, h_size=64):
        # Import Keras here to avoid messing with multiprocessing context too early
        from keras.layers import Dense, Dropout, LeakyReLU
        from keras.models import Sequential

        input_dim = np.prod(obs_shape) + np.prod(act_shape)

        self.model = Sequential()
        self.model.add(Dense(int(h_size), input_dim=int(input_dim)))
        self.model.add(LeakyReLU())

        self.model.add(Dropout(0.5))
        self.model.add(Dense(h_size))
        self.model.add(LeakyReLU())

        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))

    def run(self, obs, act):
        #print(obs)
        #print(act)
        #flat_obs = keras.layers.Flatten(obs)
        flat_obs = tf.contrib.layers.flatten(obs)
        x = tf.concat([flat_obs, act], axis=1)
        return self.model(x)

class SimpleConvolveObservationQNet(FullyConnectedMLP):
    """
    Network that has two convolution steps on the observation space before flattening,
    concatinating the action and being an MLP.
    """

    def __init__(self, obs_shape, act_shape, h_size=64):
        after_convolve_shape = (
            int(ceil(ceil(obs_shape[0] / 4) / 3)),
            int(ceil(ceil(obs_shape[1] / 4) / 3)),
            8)
        super().__init__(after_convolve_shape, act_shape, h_size)

    def run(self, obs, act):
        if len(obs.shape) == 3:
            # Need to add channels
            obs = tf.expand_dims(obs, axis=-1)
        # Parameters taken from GA3C NetworkVP
        c1 = tf.layers.conv2d(obs, 4, kernel_size=8, strides=4, padding="same", activation=tf.nn.relu)
        c2 = tf.layers.conv2d(c1, 8, kernel_size=6, strides=3, padding="same", activation=tf.nn.relu)
        return super().run(c2, act)
