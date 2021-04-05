import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LeakyReLU, Softmax, Input, ThresholdedReLU, Flatten
from tensorflow.keras.activations import sigmoid
from matplotlib import pyplot as plt
from util import *
import os
from models import *
# def quantization_evaluation(model, granuality = 0.001, k=2, saveImg = False, name="", bitstring=True):
#     dim_num = int(1/granuality) + 1
#     count = np.arange(0, 1 + granuality, granuality)
#     output = np.zeros((dim_num, dim_num))
#     input = np.zeros((dim_num*dim_num, 2))
#     line = 0
#     for i in range(count.shape[0]):
#         for j in range(count.shape[0]):
#             input[line, 0] = count[i]
#             input[line, 1] = count[j]
#             line = line + 1
#     channel_data = tf.constant(input)
#     channel_label = tf.math.argmax(channel_data, axis=1)
#     if bitstring:
#         channel_data = float_to_floatbits(channel_data)
#     ds = Dataset.from_tensor_slices((channel_data, channel_label)).batch(dim_num*dim_num)
#     for features, labels in ds:
#         # features_mod = tf.ones((1, 1))
#         # features_mod = tf.concat((features_mod, tf.reshape(features, (1, features.shape[1]))), axis=1)
#         features_mod = tf.ones((features.shape[0], features.shape[1], 1)) * 1
#         features_mod = tf.concat((features_mod, features), axis=2)
#         out = model(features_mod)
#         prediction = tf.argmax(out, axis=1)
#     line = 0
#     for i in range(count.shape[0]):
#         for j in range(count.shape[0]):
#             if prediction[line] == labels[line]:
#                 output[i, j] = 1
#             line = line + 1
#     if saveImg == False:
#         plot_quantization_square(output, granuality)
#     else:
#         save_quantization_square(output, granuality, name)
# def save_quantization_square(output, granuality, name):
#     dim_num = int(1 / granuality) + 1
#     count = np.arange(0, 1 + granuality, granuality)
#     output = np.flip(output, axis=0)
#     plt.imshow(output, cmap="gray")
#     step_x = int(dim_num / (5 - 1))
#     x_positions = np.arange(0, dim_num, step_x)
#     x_labels = count[::step_x]
#     y_labels = np.array([1, 0.75, 0.5, 0.25, 0])
#     plt.xticks(x_positions, x_labels)
#     plt.yticks(x_positions, y_labels)
#     plt.savefig(name)
# def plot_quantization_square(output, granuality):
#     dim_num = int(1 / granuality) + 1
#     count = np.arange(0, 1 + granuality, granuality)
#     output = np.flip(output, axis=0)
#     plt.imshow(output, cmap="gray")
#     step_x = int(dim_num / (5 - 1))
#     x_positions = np.arange(0, dim_num, step_x)
#     x_labels = count[::step_x]
#     y_labels = np.array([1, 0.75, 0.5, 0.25, 0])
#     plt.xticks(x_positions, x_labels)
#     plt.yticks(x_positions, y_labels)
#     plt.show()
def optimal_model(granuality=0.001):
    dim_num = int(1 / granuality) + 1
    count = np.arange(0, 1 + granuality, granuality)
    output = np.ones((dim_num, dim_num))
    line = 0
    for i in range(0, dim_num):
        for j in range(0, dim_num):
            if count[i] >= 2/3:
                quant1 = 1
            else:
                quant1 = 0
            if count[j] >= 1/3:
                quant2 = 1
            else:
                quant2 = 0
            if quant1 == quant2 and quant1 == 0:
                if count[i] < count[j]:
                    output[i, j] = 0
            elif quant1 == quant2 and quant2 == 1:
                if count[i] < count[j]:
                    output[i, j] = 0
            elif quant2 > quant1 and count[j] < count[i]:
                output[i, j] = 0
    plot_quantization_square(output, granuality)
def get_num_from_binary(binary):
    power = 0
    sum = 0
    for i in range(binary.shape[0]-1, -1, -1):
        if binary[i] > 0:
            sum = sum + 2**power
        power = power + 1
    return sum
def quantization_evaluation_regression(model, granuality = 0.0001):
    count = np.arange(0, 1, 0.001)
    out = np.zeros(count.shape)
    j = 0
    for i in count:
        data_set = tf.ones((1,1)) * i
        # input = float_to_floatbits(data_set)
        # input = tf.concat((tf.reshape(data_set, (1,1)), input), axis=1)
        tim = model(data_set)
        out[j] = tim[0, 0]
        j += 1
    plt.plot(count, out)
    plt.xlabel("input")
    plt.ylabel("output")
    plt.show()
def variance_graph(model, N = 1, k=2, bitstring=True):
    # tp_fn = ExpectedThroughput(name = "throughput")
    tp_fn = TargetThroughput(name = "target throughput", bit_string=bitstring)
    # tp_fn = ExpectedThroughput(name = "target throughput")
    a_fn = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    # tp_fn = Regression_ExpectedThroughput()
    # a_fn = Regression_Accuracy()
    result = np.zeros((N,2))
    acc = np.zeros((N, ))
    tf.random.set_seed(80)
    print("Testing Starts")
    for e in range(0, N):
        tp_fn.reset_states()
        a_fn.reset_states()
        if bitstring:
            ds = gen_channel_quality_data_float_encoded(10000, k=k)
        else:
            ds = gen_channel_quality_data_float_encoded(10000, k=k)
        for features, labels in ds:
            # prediction = tf.reshape(model(features), (1,))
            if bitstring:
                features_mod = tf.ones((features.shape[0], features.shape[1], 1)) * 1
                features_mod = tf.concat((features_mod, features), axis=2)
            else:
                print(features.shape)
                # features_mod = tf.ones((features.shape[0], features.shape[1], 1), dtype=tf.float32)
                # print(features_mod.shape)
                # features_mod = tf.concat(
                #     (features_mod, tf.reshape(features, (features.shape[0], features.shape[1], 1))), axis=2)
            prediction = model(features)
            prediction = prediction[:, :k]
            # tp_fn(labels, prediction, features)
            a_fn(labels, prediction)
            # if a_fn.result() != 1:
            #     print(features, '\n', prediction)
            # if e%1000 == 0:
            #     template = "At Test case {}, The The maximum Throughput is {}, Expected Throughout is {}, the Accuracy is {}"
            #     print(template.format(e, tp_fn.result()[0], tp_fn.result()[1], a_fn.result()))
        acc[e] = a_fn.result()
        result[e, 0] = tp_fn.result()[0]
        result[e, 1] = tp_fn.result()[1]
    print("Max Throughput:", np.nanmean(result[:, 0]))
    print("Max Throughput Variance:", np.nanvar(result[:, 0]))
    print("Expected Throughput:", np.nanmean(result[:, 1]))
    print("Expected Throughput Variance:", np.nanvar(result[:, 1]))
    print("Accuracy:", np.nanmean(acc))
    print("Accuracy Variance:", np.nanvar(acc))
    return np.nanmean(acc), np.nanmean(result[:, 0]), np.nanmean(result[:, 1])
def variance_graph_accuracy(model, N = 10000):
    # tp_fn = ExpectedThroughput(name = "throughput")
    # tp_fn = ExpectedThroughput(name = "target throughput")
    # a_fn = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    a_fn = tf.keras.metrics.MeanSquaredError()
    # tp_fn = Regression_ExpectedThroughput()
    # a_fn = Regression_Accuracy()
    result = np.zeros((N,2))
    acc = np.zeros((N, ))
    tf.random.set_seed(80)
    print("Testing Starts")
    for e in range(0, N):
        a_fn.reset_states()
        # ds = gen_data(1, k=10, batchsize=1)
        ds = gen_regression_data(N=1, batchsize=1, reduncancy=9)
        for features, labels in ds:
            # prediction = tf.reshape(model(features), (1,))
            features_mod = tf.ones((features.shape[0], 1)) * e
            features_mod = tf.concat((features_mod, features), axis=1)
            prediction = model(features_mod)
            a_fn(labels, prediction)
            # if a_fn.result() != 1:
            #     print(features, '\n', prediction)
            # if e%1000 == 0:
            #     template = "At Test case {}, The The maximum Throughput is {}, Expected Throughout is {}, the Accuracy is {}"
            #     print(template.format(e, tp_fn.result()[0], tp_fn.result()[1], a_fn.result()))
        acc[e] = a_fn.result()
    print("Accuracy:", np.nanmean(acc))
    print("Accuracy Variance:", np.nanvar(acc))
def three_run_avg(model, N=3000):
    tf.random.set_seed(80)
    tp_fn = ExpectedThroughput(name="throughput")
    a_fn = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    result = np.zeros((N, 2))
    acc = np.zeros((N,))
    tf.random.set_seed(80)
    print("Testing Starts")
    for e in range(0, N):
        tp_fn.reset_states()
        a_fn.reset_states()
        ds = gen_data(1, k=10, batchsize=1)
        for features, labels in ds:
            features_r = tf.reverse(features, axis=[1])
            features_split = tf.concat((features[:, 5:, ], features[:, 0: 5]), axis=1)
            prediction = Softmax()(model(features))
            prediction_r = tf.reverse(Softmax()(model(features_r)), axis=[1])
            prediction_split = Softmax()(model(features_split))

            # prediction = Softmax()(model(ranking_transform(features)))
            # prediction_r = tf.reverse(Softmax()(model(ranking_transform(features_r))), axis=[1])
            # prediction_split = Softmax()(model(ranking_transform(features_split)))

            prediction_split = tf.concat((prediction_split[:, 5:, ], prediction_split[:, 0: 5]), axis=1)
            prediction = (prediction + prediction_r + prediction_split)/3
            tp_fn(labels, prediction, features)
            a_fn(labels, prediction)
            # if a_fn.result() != 1:
            #     print(features, '\n', prediction)
            if e % 1000 == 0:
                template = "At Test case {}, The The maximum Throughput is {}, Expected Throughout is {}, the Accuracy is {}"
                print(template.format(e, tp_fn.result()[0], tp_fn.result()[1], a_fn.result()))
        result[e, 0] = tp_fn.result()[0]
        result[e, 1] = tp_fn.result()[1]
        acc[e] = a_fn.result()
    print("Max Throughput:", result[:, 0].mean())
    print("Max Throughput Variance:", result[:, 0].var())
    print("Expected Throughput:", result[:, 1].mean())
    print("Expected Throughput Variance:", result[:, 1].var())
    print("Accuracy:", acc.mean())
    print("Accuracy Variance:", acc.var())
def plot_data(arr):
    cut = 0
    for i in range(20, arr.shape[0]):
        if arr[i, 0] == 0:
            cut = i
            break
    arr = arr[:i, :]
    x = np.arange(0, arr.shape[0])
    error = arr[0]
    # plt.plot(x, arr[:,3])
    # plt.plot(x, arr[:,7])
    # plt.plot(x, arr[:,2])
    plt.plot(x, arr[:, 1])
    # plt.plot(x, arr[:, 3])
    plt.title("Accuracy")
    # plt.legend(("Training", "Test", "Maximum"))
    # plt.legend(("loss"))
    # plt.legend(("Quantization Level Count"))
    plt.show()
def check_multiple_models(dir_name):
    list = os.listdir(dir_name)
    list.sort()
    acc_list = []
    throughput = []
    max_throughput = []
    max_file_name = 0
    max_acc = -1
    for item in list:
        if item[-3:] == ".h5":
            model = tf.keras.models.load_model(dir_name + item)
            print(model.summary())
            res = variance_graph(model, bitstring=True)
            acc_list.append(res[0])
            throughput.append(res[2])
            max_throughput.append(res[1])
            if res[0] > max_acc:
                max_acc = res[0]
                max_file_name = item
    throughput = np.array(throughput)
    acc_list = np.array(acc_list)
    max_throughput = np.array(max_throughput)
    print("the best model is ",np.argmax(acc_list))
    print("the max accuracy is ", np.max(acc_list))
    print("the min accuracy is ", np.min(acc_list))
    print("the mean accuracy is", np.mean(acc_list))
    print("the max throughput (upper bound) is", np.mean(max_throughput))
    print("the max throughput is", np.max(throughput))
    print("the max throughput is", np.min(throughput))
    print("the max throughput is", np.mean(throughput))
    print(dir_name + max_file_name)
    bestmodel = tf.keras.models.load_model(dir_name + max_file_name)
    quantization_evaluation(bestmodel, granuality=0.01, bitstring=True)
    return
if __name__ == "__main__":
    custome_obj = {'Closest_embedding_layer' : Closest_embedding_layer}
    model = tf.keras.models.load_model("trained_models/Jul 23rd/VAE quantization scheduling k=2,L=10.h5", custom_objects=custome_obj)
    # quantization_evaluation_regression(model)
    quantization_evaluation(model, granuality=0.01, bitstring=False)
    variance_graph(model, N=1, k=30, bitstring=False)
    training_data = np.load("trained_models/Jul 23rd/VAE quantization scheduling k=30,L=40.npy")
    plot_data(training_data)

    A[2]
    check_multiple_models("./trained_models/Jul 8th/k=2 distinct regression network/")
    A[2]
    file = "trained_models/Jul 8th/k=2 regression network/2_user_1_qbit_threshold_encoder_tanh(relu)_seed=5"
    # file = "trained_models/Sept 25/k=2, L=2/Data_gen_encoder_L=1_k=2_tanh_annealing"
    model_path = file + ".h5"
    training_data_path = file + ".npy"
    # training_data_path1 = file + "_cont.npy"
    # training_data_path2 = file + "_cont2.npy"
    # training_data = np.concatenate((np.load(training_data_path), np.load(training_data_path1)), axis=0)
    training_data = np.load(training_data_path)
    model = tf.keras.models.load_model(model_path)
    # print(model.summary())
    # model = create_uniformed_quantization_model(k=2, bin_num=2)
    plot_data(training_data)
    A[2]
    # optimal_model()
    variance_graph(model, N=1, k=30)
    # variance_graph_accuracy(model, N=1000)
    quantization_evaluation(model)
    # quantization_evaluation_regression(model)

