import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LeakyReLU, Softmax, Input, ThresholdedReLU, Flatten
from tensorflow.keras.activations import sigmoid
import random
from util import *
from models import *
def train_step(features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        # predictions = model(ranking_transform(features))
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_throughput(labels, predictions, features)
    train_accuracy(labels, predictions)
def test_step(features, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(features)
    # predictions = model(ranking_transform(features))
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)
    test_throughput(labels, predictions, features)
if __name__ == "__main__":
    N = 10000
    k = 10
    EPOCHS = 2000
    tf.random.set_seed(80)
    graphing_data = np.zeros((EPOCHS, 8))
    model0 = tf.keras.models.load_model("trained_models/Sept 19th/N_10000_5_Layer_MLP_regression.h5")
    model1 = create_regression_MLP_netowkr((k,), k)
    model = boosting_regression_model([model0, model1],[k, 1], k)
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_throughput = Regression_ExpectedThroughput(name='train_throughput')
    train_accuracy = Regression_Accuracy()
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_throughput = Regression_ExpectedThroughput(name='test_throughput')
    test_accuracy = Regression_Accuracy()
    train_ds = gen_data(N, k, 0, N)
    test_ds = gen_data(100, k, 0, 1)
    current_acc = 0
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_ds = gen_data(N, k, 0, N)
        train_loss.reset_states()
        train_throughput.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        test_throughput.reset_states()
        for features, labels in train_ds:
            train_step(features, labels)
        for features, labels in test_ds:
            test_step(features, labels)
        template = 'Epoch {}, Loss: {}, Accuracy:{}, max: {}, expected:{}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result(),
                              train_throughput.result()[0],
                              train_throughput.result()[1],
                              test_loss.result(),
                              test_accuracy.result()))
        graphing_data[epoch, 0] = train_loss.result()
        graphing_data[epoch, 1] = train_accuracy.result()
        graphing_data[epoch, 2] = train_throughput.result()[0]
        graphing_data[epoch, 3] = train_throughput.result()[1]
        graphing_data[epoch, 4] = test_loss.result()
        graphing_data[epoch, 5] = test_accuracy.result()
        graphing_data[epoch, 6] = test_throughput.result()[0]
        graphing_data[epoch, 7] = test_throughput.result()[1]
    fname_template = "./trained_models/Sept 22_23/N_{}_boosting_regression_1{}"
    fname_template = "./trained_models/Sept 22_23/Data_gen_boosting_regression_1{}"
    np.save(fname_template.format(".npy"), graphing_data)
    model.save(fname_template.format(N, ".h5"))