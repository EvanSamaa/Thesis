import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LeakyReLU, Softmax, Input, ThresholdedReLU, Flatten
from tensorflow.keras.activations import sigmoid
from util import *
from BinaryEncodingModel import *
from models import *

def freeze_encoder_layers(model):
    for layer in model.layers:
        if layer.name[0:7] == "encoder":
            layer.trainable = False
        elif layer.name[0:7] == "decoder":
            layer.trainable = True
    return model
def freeze_decoder_layers(model):
    for layer in model.layers:
        if layer.name[0:7] == "decoder":
            layer.trainable = False
        elif layer.name[0:7] == "encoder":
            layer.trainable = True
    return model
def unfreeze_all(model):
    for layer in model.layers:
        layer.trainable = True
    return model
def train_step(features, labels, N=None, encode_only=False):
    if N != None:
        return train_step_with_annealing(features, labels, N, encode_only)
    with tf.GradientTape() as tape:
        predictions = model(features)
        logits = predictions[:, :k]
        vae_loss = vector_quantization_loss(predictions)
        ce_loss = classification_loss(labels, logits)
        # ce_loss = regression_loss(labels, logits)
        loss = ce_loss + vae_loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(ce_loss)
    train_accuracy(labels, logits)
    # train_accuracy(vae_loss)
def test_step(features, labels, N=None):
    if N != None:
        return test_step_with_annealing(features, labels, N)
    predictions = model(features)
    logits = predictions[:, :k]
    vae_loss = vector_quantization_loss(predictions)
    ce_loss = classification_loss(labels, logits)
    # ce_loss = regression_loss(labels, logits)
    t_loss = ce_loss + vae_loss
    test_loss(t_loss)
    # test_accuracy(labels, logits)
    test_accuracy(t_loss)
    # test_throughput(labels, predictions, features)
def train_step_with_annealing(features, labels, N, encode_only=False):
    features_mod = tf.ones((features.shape[0], 1)) * N
    features_mod = tf.concat((features_mod, features), axis=1)
    # print(feature)
    if encode_only == False:
        with tf.GradientTape() as tape:
            predictions = model(features_mod)
            # predictions = model(ranking_transform(features))
            quantization = submodel(features_mod)[0:1000, :]
            quantization0 = quantization + tf.stop_gradient(tf.sign(quantization) - quantization)
            loss = loss_object(labels, predictions)
            # quantization0 = tf.reshape(quantization, (1, 1000, 3))
            # loss = -loss_model(quantization0)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    gradient_record(tf.reduce_mean(tf.square(gradients[0])))
    quantization_count(Quantization_count(tf.sign(quantization)))
    # train_throughput(labels, predictions, features)
    train_accuracy(labels, predictions)
def test_step_with_annealing(features, labels, N):
    features_mod = tf.ones((features.shape[0], 1)) * N
    features_mod = tf.concat((features_mod, features), axis=1)
    predictions = model(features_mod)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)
    # test_throughput(labels, predictions, features)
if __name__ == "__main__":
    # test_model()
    # A[2]
    fname_template_template = "./trained_models/Jul 23rd/VAE quantization scheduling k=30,L=3_{}"
    N = 5000
    k = 30
    L = 3
    switch = 20
    EPOCHS = 20000
    code_size = 4
    for seed in range(0, 1):
        tf.keras.backend.clear_session()
        fname_template = fname_template_template.format(seed) + "{}"
        tf.random.set_seed(seed)
        graphing_data = np.zeros((EPOCHS, 8))
        # model = Recover_uniform_quantization(input_shape = [24,], L=3)
        model = DiscreteVAE(k, L, (k, ), code_size)
        model = DiscreteVAE_diff_scheduler(k, L, (k, ), code_size)
        # model = DiscreteVAE_regression(L, (k, ), code_size)
        classification_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        regression_loss = tf.keras.losses.MeanSquaredError()
        vector_quantization_loss = VAE_encoding_loss(k, code_size)

        optimizer = tf.keras.optimizers.Adam(lr=0.0001)
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")
        # train_accuracy = tf.keras.metrics.Mean(name='train_loss')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        # test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_acc")
        test_accuracy = tf.keras.metrics.Mean(name='train_loss')
        quantization_count = tf.keras.metrics.Mean(name='test_loss')
        # test_ds = gen_encoding_data(N=100, Sequence_length=1000, batchsize=100)
        # test_ds = gen_regression_data(N=1000, batchsize=1000, reduncancy=1)
        test_ds = gen_channel_quality_data_float_encoded(100, k)
        min_loss = -100
        encode_onlyy = False
        for epoch in range(EPOCHS):
            # Reset the metrics at the start of the next epoch
            # train_ds = gen_encoding_data(N=1000, Sequence_length=1000, batchsize=1000)
            # train_ds = gen_regression_data(reduncancy=1)
            train_ds = gen_channel_quality_data_float_encoded(N, k)
            train_loss.reset_states()
            quantization_count.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()
            for features, labels in train_ds:
                train_step(features, labels)
                # train_step(features, labels, epoch, encode_onlyy)
            for features, labels in test_ds:
                test_step(features, labels)
                # test_step(features, labels, epoch)
            template = 'Epoch {}, Loss: {}, Accuracy:{}, Test Loss: {}, Test Accuracy: {}'
            print(template.format(epoch + 1,
                                  train_loss.result(),
                                  train_accuracy.result(),
                                  test_loss.result(),
                                  test_accuracy.result()))
            graphing_data[epoch, 0] = train_loss.result()
            graphing_data[epoch, 1] = train_accuracy.result()
            graphing_data[epoch, 4] = test_loss.result()
            graphing_data[epoch, 5] = test_accuracy.result()
            if train_accuracy.result() > min_loss:
                # model = replace_tanh_with_sign(model, binary_encoding_model_regularization, k=8)
                model.save(fname_template.format(".h5"))
                min_loss = train_accuracy.result()
            if epoch%500 == 0:
                # print(model.get_layer("closest_embedding_layer").E)
                if epoch >= 1000:
                    improvement = graphing_data[epoch-1000: epoch-500, 0].mean() - graphing_data[epoch-500: epoch, 0].mean()
                    print("the accuracy improvement in the past 500 epochs is ", improvement)
                    if improvement <= 0.000001:
                        break
        np.save(fname_template.format(".npy"), graphing_data)
        # fname_template = "~/quantization_communication/trained_models/Sept 25th/Data_gen_encoder_L10_hard_tanh{}"
    print("Training data end ")


