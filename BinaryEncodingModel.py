import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LeakyReLU, Softmax, Input, ThresholdedReLU, Flatten
from tensorflow.keras.activations import sigmoid
import random
from util import *

def Recover_uniform_quantization(input_shape=(5,), L=3):
    inputs = Input(shape=input_shape)
    epochs = inputs[:, 1][0]
    inputs_mod = inputs[:, 1:]
    x = Encoder_module_annealing(L)(inputs_mod, epochs)
    x = Dense(30)(x)
    x = LeakyReLU()(x)
    x = Dense(1)(x)
    model = Model(inputs, x, name="Recover uniform Quantization")
    print(model.summary())
    return model

def Encoder_module_annealing(L, i=0):
    def encoder_module(x, N):
        x = Dense(100, name="encoder_dense_1_{}".format(i))(x)
        x = LeakyReLU()(x)
        # x = Dense(20, name="encoder_dense_2_{}".format(i))(x)
        # x = LeakyReLU()(x)
        x = Dense(L, name="encoder_dense_3_{}".format(i))(x)
        # x = sign_relu_STE(x)
        # x = tf.tanh(tf.keras.layers.ReLU()(x), name="tanh_pos") + tf.stop_gradient(binary_activation(x) - tf.tanh(tf.keras.layers.ReLU()(x)))
        x = annealing_tanh(x, N, "tanh_pos") + tf.stop_gradient(tf.math.sign(x) - annealing_tanh(x, N, "tanh_neg"))
        # x = tf.tanh(tf.keras.layers.ReLU()(x), name="tanh_pos")
        # x = tf.tanh(tf.keras.layers.ReLU()(x))
        # x = hard_tanh(x) + tf.stop_gradient(tf.sign(x) - hard_tanh(x))
        return x
    return encoder_module
def Encoder_module_regularization(L, i=0, saved=False):
    def encoder_module(x):
        x = Dense(50, name="encoder_dense_1_{}".format(i))(x)
        x = LeakyReLU()(x)
        x = Dense(80, name="encoder_dense_2_{}".format(i))(x)
        x = LeakyReLU()(x)
        x = Dense(L, name="encoder_dense_3_{}".format(i))(x)
        if saved == False:
            x = tf.keras.activations.tanh(x)
        else:
            x = tf.sign(x)
        return x
    return encoder_module

def binary_encoding_model(input_shape, k):
    inputs = Input(shape=input_shape)
    epochs = inputs[:, 1][0]
    inputs_mod = inputs[:, 1:]
    # x_list = tf.split(inputs_mod, num_or_size_splits=k, axis=1)
    x = Encoder_module_annealing(3)(inputs_mod, epochs)
    x = Dense(30, name="decoder_dense_1")(x)
    x = LeakyReLU()(x)
    x = Dense(20, name="decoder_dense_2")(x)
    x = LeakyReLU()(x)
    x = Dense(20, name="decoder_dense_3")(x)
    x = LeakyReLU()(x)
    out = Dense(k, name="decoder_dense_4")(x)
    model = Model(inputs, out, name = "binary encoding model")
    print(model.summary())
    return model

def binary_encoding_model_regularization(input_shape, k, saved=False):
    inputs = Input(shape=input_shape)
    x = Encoder_module_regularization(3, saved=saved)(inputs)
    x = Dense(30, name="decoder_dense_1")(x)
    x = LeakyReLU()(x)
    x = Dense(20, name="decoder_dense_2")(x)
    x = LeakyReLU()(x)
    out = Dense(8, name="decoder_dense_3")(x)
    model = Model(inputs, out, name="binary encoding model")
    print(model.summary())
    return model

def binary_encoding_model_neg_sampling(input_shape, k):
    inputs = Input(shape=input_shape)
    epochs = inputs[:, 1][0]
    inputs_mod = inputs[:, 1:]
    embedding = Encoder_module_annealing(3)(inputs_mod, epochs)
