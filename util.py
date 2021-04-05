import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LeakyReLU, Softmax, Input, ThresholdedReLU, Flatten
from tensorflow.keras.activations import sigmoid
import random
import math
from itertools import combinations
import tensorflow.keras.backend as KB
# import cv2
import os
from soft_sort.tf_ops import soft_rank
import scipy as sp
from generate_batch_data import generate_batch_data, generate_batch_data_with_angle
# from models import k_clustering_hieristic
# from matplotlib import pyplot as plt
# ==========================  Data gen ============================s
import tensorflow as tf

class ModelTrainer():
    def __init__(self, save_dir, data_cols=2, epoch=100000):
        self.save_dir = save_dir
        self.data = np.zeros((epoch, data_cols))
        self.data_cols = data_cols
    def log(self, epoch, vals):
        for i in range(self.data_cols):
            self.data[epoch, i] = vals[i]
    def save(self):
        self.data
        np.save(self.save_dir, self.data)
class Weighted_sumrate_model():
    def __init__(self, K, M, N_rf, N, alpha:float, hard_decision = True, loss_fn=None):
        if (loss_fn==None):
            self.lossfn = Sum_rate_utility_WeiCui_seperate_user(K, M, 1) # loss function to calculate sumrate
        else:
            self.lossfn = loss_fn
        self.time = 0 # record time stamp
        self.alpha = alpha  # this determines the decay rate of the weighted sum
                            # small alpha means low delay rate, i.e. the model
        self.hard_decision = hard_decision # determine whether to harden the decision vector or not (for training)
        self.N_rf = N_rf # Nrf of this
        self.K = K
        self.M = M
        self.N = N
        # structure for saving data
        record_shape = (1, N, K)
        self.rates = np.zeros(record_shape, dtype=np.float32) # keep the cumulative rates from the past timestamp
        self.weighted_rates = np.zeros(record_shape, dtype=np.float32)
        self.decisions = np.zeros([])
    def reset(self):
        record_shape = (1, self.N, self.K)
        self.time = 0
        self.rates = np.zeros(record_shape, dtype=np.float32)  # keep the cumulative rates from the past timestamp
        self.weighted_rates = np.zeros(record_shape, dtype=np.float32)
    def get_rates(self):
        # the rates
        return -self.rates
    def get_weighted_rates(self):
        # the rates
        return -self.weighted_rates
    def compute_weighted_loss(self, X, G, weight=None, update=True):
        # this function assumes the caller will feed in the soft decision vector
        # must call increment before this step to obtain the correct loss and make the correct update
        # this will simply compute a loss, without applying the weighted sumrate rule

        ###########################################   R_t is alwayus positive, loss function returns positive too  ###############################################
        local_X = X
        if self.hard_decision:
            local_X = X + tf.stop_gradient(Harden_scheduling_user_constrained(self.N_rf, self.K, self.M)(X) - X)
        R_t = self.lossfn(local_X, G)
        if self.time == 0:
            R_t_bar = R_t
        else:
            if weight is None:
                weight = self.get_weight()
                R_t_bar = (1.0 - self.alpha) * self.rates[-2] + self.alpha * R_t
            else:
                # if you are not
                rate = tf.math.log(1/weight)
                R_t_bar = (1.0 - self.alpha) * rate + self.alpha * R_t
        if update:
            self.rates[-1] = R_t_bar
            self.weighted_rates[-1] = R_t * weight
        return -tf.reduce_sum(R_t * weight, axis=1)
    def get_weight(self):
        # this function assumes the caller will feed in the soft decision vector
        # must call increment before this step to obtain the correct weight
        if self.time == 0:
            return np.ones((self.N, self.K), dtype=np.float32)
        weight = np.array(1.0/np.exp(self.rates[-2, :, :]), np.float32)
        return weight
    def get_binary_weights(self):
        # this function assumes the caller will feed in the soft decision vector
        # this will simply compute a loss, without applying the weighted sumrate rule
        return np.array(np.where(self.get_weight() > 0.5, 1.0, 0.0), np.float32)
    def compute_raw_loss(self, X, G):
        # this function assumes the caller will feed in the soft decision vector
        # this will simply compute a loss, without applying the weighted sumrate rule
        local_X = X
        if self.hard_decision:
            local_X = Harden_scheduling(self.N_rf, self.K, self.M)(X)
        R_t = self.lossfn(local_X, G)
        return -tf.reduce_sum(R_t, axis=1)
    def plot_average_rates(self, show=True):
        from matplotlib import pyplot as plt
        rate = tf.reduce_sum(self.rates, axis=2)
        rate = -tf.reduce_mean(rate, axis=1)
        plt.plot(rate)
        plt.title("weighted sumrate over time")
        plt.xlabel("episodes")
        plt.ylabel("Weighted Sumrate")
        if show:
             plt.show()
    def plot_activation(self, show=True):
        from matplotlib import pyplot as plt
        rate = tf.reduce_sum(self.rates, axis=0)
        rate = tf.reduce_sum(rate, axis=0)
        plt.bar(np.arange(0, self.rates.shape[2]), rate)
        plt.show()
    def increment(self):
        self.time = self.time + 1
        self.rates = np.concatenate([self.rates, np.zeros([1, self.rates.shape[1], self.rates.shape[2]])], axis=0)
        self.weighted_rates = np.concatenate([self.weighted_rates, np.zeros([1, self.weighted_rates.shape[1], self.weighted_rates.shape[2]])], axis=0)

def rebar_loss(logits, Nrf, M, K):
    # logit shape = (N, passes, M*K, N_rf)
    epsilon = 1E-12
    u = tf.random.uniform(z.shape.as_list(), dtype=z.dtype)
    gumbel = - tf.math.log(-tf.math.log(u + epsilon) + epsilon, name="gumbel")
    z = gumbel + u

    def truncated_gumbel(gumbel, truncation):
        return -tf.math.log(epsilon + tf.exp(-gumbel) + tf.exp(-truncation))
    v = tf.random.uniform(logits.shape.as_list(), dtype=logits.dtype)
    gumbel = -tf.math.log(-tf.math.log(v + epsilon) + epsilon, name="gumbel")
    topgumbels = gumbel + tf.reduce_logsumexp(logits, axis=-1, keepdims=True)
    topgumbel = tf.reduce_sum(s*topgumbels, axis=-1, keepdims=True)
    z_hat = truncated_gumbel(gumbel + logits, topgumbel)


def generate_link_channel_data_fullAOE(N, K, M, Nrf, SNR=20, sigma2_h=0.1, sigma2_n=0.1):
    # for all previously trained models, SNR=20
    Lp = 2  # Number of Paths
    P = tf.constant(sp.linalg.dft(M), dtype=tf.complex64) # DFT matrix
    P = P/tf.sqrt(tf.constant(M, dtype=tf.complex64))/tf.sqrt(tf.constant(Nrf, dtype=tf.complex64))*tf.sqrt(tf.constant(10**(SNR/10), dtype=tf.complex64))
    P = tf.expand_dims(P, 0)
    P = tf.tile(P, (N, 1, 1))
    LSF_UE = np.array([0.0, 0.0], dtype=np.float32)  # Mean of path gains
    Mainlobe_UE = np.array([0, 0], dtype=np.float32)  # Mean of the AoD range
    HalfBW_UE = np.array([180.0, 180.0], dtype=np.float32)  # Half of the AoD range
    h_act_batch = tf.constant(generate_batch_data(N, M, K, Lp, LSF_UE, Mainlobe_UE, HalfBW_UE), dtype=tf.complex64)
    # taking hermecian
    h_act_batch = tf.transpose(h_act_batch, perm=(0, 2, 1), conjugate=True)
    G = tf.matmul(h_act_batch, P)
    # noise = tf.complex(tf.random.normal(G.shape, 0, sigma2_n, dtype=tf.float32),
    #                    tf.random.normal(G.shape, 0, sigma2_n, dtype=tf.float32))
    G_hat = G
    return G_hat
def generate_link_channel_data(N, K, M, Nrf, sigma2_h=0.1, sigma2_n=0.1):
    Lp = 2  # Number of Paths
    P = tf.constant(sp.linalg.dft(M), dtype=tf.complex64) # DFT matrix
    P = P/tf.sqrt(tf.constant(M, dtype=tf.complex64))/tf.sqrt(tf.constant(Nrf, dtype=tf.complex64))*tf.sqrt(tf.constant(100, dtype=tf.complex64))
    P = tf.expand_dims(P, 0)
    P = tf.tile(P, (N, 1, 1))
    LSF_UE = np.array([0.0, 0.0], dtype=np.float32)  # Mean of path gains
    Mainlobe_UE = np.array([0, 0], dtype=np.float32)  # Mean of the AoD range
    HalfBW_UE = np.array([30.0, 30.0], dtype=np.float32)  # Half of the AoD range
    h_act_batch = tf.constant(generate_batch_data(N, M, K, Lp, LSF_UE, Mainlobe_UE, HalfBW_UE), dtype=tf.complex64)
    # taking hermecian
    h_act_batch = tf.transpose(h_act_batch, perm=(0, 2, 1), conjugate=True)
    G = tf.matmul(h_act_batch, P)
    # noise = tf.complex(tf.random.normal(G.shape, 0, sigma2_n, dtype=tf.float32),
    #                    tf.random.normal(G.shape, 0, sigma2_n, dtype=tf.float32))
    G_hat = G
    return G_hat
def generate_link_channel_data_with_angle(N, K, M, sigma2_h=0.1, sigma2_n=0.1):
    Lp = 2  # Number of Paths
    P = tf.constant(sp.linalg.dft(M), dtype=tf.complex64) # DFT matrix
    P = tf.expand_dims(P, 0)
    P = tf.tile(P, (N, 1, 1))
    LSF_UE = np.array([0.0, 0.0], dtype=np.float32)  # Mean of path gains
    Mainlobe_UE = np.array([0, 0], dtype=np.float32)  # Mean of the AoD range
    HalfBW_UE = np.array([180.0, 180.0], dtype=np.float32)  # Half of the AoD range
    data, angle = generate_batch_data_with_angle(N, M, K, Lp, LSF_UE, Mainlobe_UE, HalfBW_UE)
    h_act_batch = tf.constant(data, dtype=tf.complex64)
    # taking hermecian
    h_act_batch = tf.transpose(h_act_batch, perm=(0, 2, 1), conjugate=True)
    G = tf.matmul(h_act_batch, P)
    # noise = tf.complex(tf.random.normal(G.shape, 0, sigma2_n, dtype=tf.float32),
    #                    tf.random.normal(G.shape, 0, sigma2_n, dtype=tf.float32))
    G_hat = G
    return G_hat, angle
def generate_supervised_link_channel_data(N, K, M, N_rf, sigma2_h=0.1, sigma2_n=0.1):
    G_hat = generate_link_channel_data(N, K, M, sigma2_h, sigma2_n)
    exstimated_result = top_N_rf_user_model(M, K, N_rf)(G_hat)
    # exstimated_result = k_clustering_hieristic(N_rf)(G_hat)
    dataset = Dataset.from_tensor_slices((G_hat, exstimated_result)).batch(N)
    return dataset
def gen_data(N, k, low=0, high=1, batchsize=30):
    channel_data = tf.random.uniform((N,k,1), low, high)
    channel_label = tf.math.argmax(channel_data, axis=1)
    dataset = Dataset.from_tensor_slices((channel_data, channel_label)).batch(batchsize)
    return dataset
def gen_channel_quality_data_float_encoded(N, k, low=0, high=1):
    channel_data = tf.random.uniform((N, k), low, high)
    channel_label = tf.math.argmax(channel_data, axis=1)
    # channel_data = float_to_floatbits(channel_data)
    dataset = Dataset.from_tensor_slices((channel_data, channel_label)).batch(500)
    return dataset
def gen_pathloss_batch(N, L, S, K, B2Bdist, inn_ratio, out_ratio, save_loc = ""):
    BSLoc = tf.zeros((1,))
    MULoc = 0.5 * B2Bdist * np.sqrt((np.random.uniform(inn_ratio ** 2, out_ratio ** 2, (N, K, 1)))) * np.exp(
        1j * (np.random.uniform(0, 1, (N, K, 1))) * 2 * np.pi) + BSLoc[0]
    dist = np.abs(MULoc - BSLoc[0])
    path_loss_dB = 128.1 + 37.6 * np.log10(dist)
    pathloss = np.power(10, -path_loss_dB / 20)
    if save_loc != "":
        np.save(save_loc, pathloss)
    pathloss = tf.constant(pathloss, dtype=tf.complex64)
    # from matplotlib import pyplot as plt
    # plt.plot(np.real(MULoc), np.imag(MULoc), 'bo')
    # plt.plot(0, 0, 'ro')
    # plt.show()
    return pathloss
def gen_pathloss(L, S, K, B2Bdist, inn_ratio, out_ratio, save_loc = ""):

    # % This function generates channel matrix in frequency domain
    # % for frequency-selective fading channels based on a 19-cell topology
    # %
    # % L: base-stations
    # % S: sectors per base-station
    # % K: users per sector
    # % N: frequency tones
    # % P: base-station antennas
    # % Q: mobile antennas
    # % B2Bdist: Base-station to base-station distance
    # %
    # % %% Output format (complex gain)
    # % Chn(l, s, m, t, k, p, q, n)
    # % complex gain from the l-th BS, s-th sector, p-th antenna
    # % to the kth user in the m-th BS, t-th sector, q-th antenna
    # % in n-th frequency tone
    BSLoc = tf.zeros((1, ))
    MULoc = 0.5 * B2Bdist * np.sqrt((np.random.uniform(inn_ratio**2, out_ratio**2,(K, 1)))) * np.exp(1j * (np.random.uniform(0, 1,(K, 1))) * 2 * np.pi) + BSLoc[0]
    dist = np.abs(MULoc - BSLoc[0])
    path_loss_dB = 128.1 + 37.6 * np.log10(dist)
    pathloss = np.power(10, -path_loss_dB / 20)
    if save_loc != "":
        np.save(save_loc, pathloss)
    pathloss = tf.constant(pathloss, dtype = tf.complex64)
    # from matplotlib import pyplot as plt
    # plt.plot(np.real(MULoc), np.imag(MULoc), 'bo')
    # plt.plot(0, 0, 'ro')
    # plt.show()
    return pathloss
def gen_realistic_data(space_file, N, K, M, Nrf):
    if space_file == "":
        pathloss = gen_pathloss(1, 1, K, 0.3, 0.1, 1.0)
    else:
        pathloss = tf.constant(np.load(space_file), dtype = tf.complex64)
    channel_data = generate_link_channel_data_fullAOE(N, K, M, Nrf, SNR=100, sigma2_h=0.1, sigma2_n=0.1)
    CSI = np.expand_dims(pathloss, axis=0) * channel_data
    return CSI
def gen_realistic_data_batch(N, K, M, Nrf):
    pathloss = gen_pathloss_batch(N, 1, 1, K, 0.3, 0.1, 1.0)
    channel_data = generate_link_channel_data_fullAOE(N, K, M, Nrf, SNR=100, sigma2_h=0.1, sigma2_n=0.1)
    CSI = pathloss * channel_data
    return CSI
def gen_number_data(N=10000, k = 7.5, batchsize=10000):
    channel_data_num = tf.random.uniform((N, 1), 0, k)
    channel_data_num = tf.cast(tf.round(channel_data_num), dtype=tf.int32)
    # channel_data_num = tf.round(channel_data_num)
    channel_data = tf.cast(tf.one_hot(channel_data_num, depth=math.ceil(k), on_value=1.0, off_value=0.0), tf.float32)
    channel_data = tf.reshape(channel_data, (N, math.ceil(k)))
    channel_label = channel_data_num
    dataset = Dataset.from_tensor_slices((channel_data
                                          , channel_label)).batch(batchsize)
    return dataset
def gen_encoding_data(N=1000, Sequence_length=10000, k=16, batchsize = 100, bit = 4):
    # dict_list = [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]
    dict_list = [[-1, -1, -1, -1], [-1, -1, -1, 1], [-1, -1, 1, -1], [-1, -1, 1, 1],
                 [-1, 1, -1, -1], [-1, 1, -1, 1], [-1, 1, 1, -1], [-1, 1, 1, 1],
                 [1, -1, -1, -1], [1, -1, -1, 1], [1, -1, 1, -1], [1, -1, 1, 1],
                 [1, 1, -1, -1], [1, 1, -1, 1], [1, 1, 1, -1], [1, 1, 1, 1]]
    output = np.zeros((N, Sequence_length, bit))
    channel_label = np.zeros((N, 1))
    for n in range(N):
        random.shuffle(dict_list)
        current = random.randint(1,k)
        current_encoding = np.array(dict_list[:current])
        idx = np.random.randint(0, current, size=Sequence_length)
        for m in range(Sequence_length):
            output[n, m] = current_encoding[idx[m]]
        channel_label[n] = current
    channel_label = tf.cast(channel_label, tf.float32)
    output = tf.cast(output, tf.float32)
    dataset = Dataset.from_tensor_slices((output, channel_label)).batch(batchsize)
    return dataset
def gen_regression_data(N=10000, batchsize=500, reduncancy=1):
    ################
    data_set = tf.random.uniform((N, ), 0, 1)
    label_set = data_set
    # modified_dataset = float_to_floatbits(data_set)
    ################
    # ones = tf.ones((N, reduncancy))
    # data_set = tf.random.uniform((N,1), 0, 1) # for redundancy
    # label_set = data_set
    # data_set = tf.concat((data_set, -data_set, tf.exp(data_set), tf.square(data_set)), axis=1)
    # data_set = tf.multiply(ones, data_set)
    # print(data_set)
    dataset = Dataset.from_tensor_slices((data_set, label_set)).batch(batchsize)
    return dataset
def nChoosek_bits(N, k):
    # each row is a
    count = sp.special.comb(N, k, exact=True)
    results = np.zeros((count, N))
    i = 0
    for bits in combinations(range(N), k):
        for bit in bits:
            results[i, bit] = 1
        i = i + 1
    return count, tf.constant(results, dtype=tf.float32)
def obtain_channel_distributions(N, K, M, Nrf, sigma2_h=0.1, sigma2_n=0.1):
    from matplotlib import pyplot as plt
    G = generate_link_channel_data(N, K, M, Nrf)
    plt.hist(tf.abs(G[:, 0, 0]).numpy(), bins=100)
    plt.show()
    max_G = tf.reduce_max(tf.abs(G), axis=0)
    mean_G = tf.reduce_mean(tf.abs(G), axis=0)
    std_G = tf.math.reduce_std(tf.abs(G), axis=0)
    print(max_G)
    plt.hist(tf.abs(max_G).numpy(), bins=100)
    plt.show()
    upper = mean_G + 2 * std_G
    print(upper)
    plt.hist(tf.abs(upper).numpy(), bins=100)
    plt.show()
    return max_G


# ============================  Metrics  ============================
def quantizaton_evaluation_numbers(model, granuality=0.0001, k=2):
    tsub_model = Model(inputs=model.input, outputs=model.get_layer("tf_op_layer_Sign").output)
    for i in range(0, 16):
        features = tf.ones((1,1))*i
        # features = tf.one_hot([0, 1, 2, 3, 4, 5, 6, 7], depth=8)[i]
        features_mod = tf.ones((1, 1))
        features_mod = tf.concat((features_mod, tf.reshape(features, (1, features.shape[0]))), axis=1)
        out = tsub_model(features_mod)
        out_2 = model(features_mod)
        # print(i, out, Softmax()(out_2))
        print(i, out)
class ExpectedThroughput(tf.keras.metrics.Metric):
    def __init__(self, name='expected_throughput', logit=True, **kwargs):
        super(ExpectedThroughput, self).__init__(name=name, **kwargs)
        self.max_throughput = self.add_weight(name='tp', initializer='zeros')
        self.expected_throughput = self.add_weight(name='tp2', initializer='zeros')
        self.count = self.add_weight(name='tp3', initializer='zeros')
        self.a = tf.math.log(tf.cast(2, dtype=tf.float32))
        self.logit = logit
    def update_state(self, y_true, y_pred, x):
        c_max = tf.gather(x, y_true, axis=1, batch_dims=1)
        i_max = tf.math.log(1 - 1.5*tf.math.log(1 - c_max))/self.a
        if self.logit:
            weighted_c = tf.math.multiply(Softmax()(y_pred), x)
        else:
            weighted_c = tf.math.multiply(y_pred, x)
        i_expected = tf.math.reduce_sum(tf.math.log(1 - 1.5*tf.math.log(1 - weighted_c))/self.a, axis=1)
        self.max_throughput.assign_add(tf.math.reduce_sum(i_max, axis=0))
        self.expected_throughput.assign_add(tf.math.reduce_sum(i_expected, axis=0))
        self.count.assign_add(y_true.shape[0])
    def result(self):
        return self.max_throughput/self.count, self.expected_throughput/self.count
class Regression_ExpectedThroughput(tf.keras.metrics.Metric):
    def __init__(self, name='expected_throughput', **kwargs):
        super(Regression_ExpectedThroughput, self).__init__(name=name, **kwargs)
        self.max_throughput = self.add_weight(name='tp', initializer='zeros')
        self.expected_throughput = self.add_weight(name='tp2', initializer='zeros')
        self.count = self.add_weight(name='tp3', initializer='zeros')
        self.a = tf.math.log(tf.cast(2, dtype=tf.float32))
    def update_state(self, y_true, y_pred, x):
        c_max = tf.gather(x, y_true, axis=1, batch_dims=1)
        i_max = tf.math.log(1 - 1.5*tf.math.log(1 - c_max))/self.a
        y_pred_rounded = tf.cast(tf.math.round(y_pred), tf.int32)
        y_pred_rounded = tf.where(y_pred_rounded >= x.shape[1], x.shape[1]-1, y_pred_rounded)
        y_pred_rounded = tf.where(y_pred_rounded < 0, 0, y_pred_rounded)
        c_pred = tf.gather(x, y_pred_rounded, axis=1, batch_dims=1)
        i_pred = tf.math.log(1 - 1.5 * tf.math.log(1 - c_pred)) / self.a
        self.max_throughput.assign_add(tf.math.reduce_sum(i_max, axis=0))
        self.expected_throughput.assign_add(tf.math.reduce_sum(i_pred, axis=0))
        self.count.assign_add(y_true.shape[0])
    def result(self):
        return self.max_throughput/self.count, self.expected_throughput/self.count
class Regression_Accuracy(tf.keras.metrics.Metric):
    def __init__(self, name='expected_throughput', **kwargs):
        super(Regression_Accuracy, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='tp', initializer='zeros')
        self.count = self.add_weight(name='tp2', initializer='zeros')
    def update_state(self, y_true, y_pred):
        y_rounded_pred = tf.cast(tf.math.round(y_pred), tf.float32)
        diff = y_true - y_rounded_pred
        count = tf.cast(tf.reduce_sum(tf.where(diff == 0, 1, 0)), tf.float32)
        # i_pred = tf.reshape((tf.math.log(1 - 1.5*tf.math.log(1 - c_pred))/self.a), (i_max.shape[0], ))
        self.correct.assign_add(count)
        self.count.assign_add(y_true.shape[0])
    def result(self):
        return self.correct/self.count
def Quantization_count(x):
    # pass in (N, encoding_size) array to delete duplicate on the first axis
    x_shape = tf.shape(x)  # (3,2)
    x1 = tf.tile(x, [1, x_shape[0]])  # [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]
    x2 = tf.tile(x, [x_shape[0], 1])  # [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]

    x1_2 = tf.reshape(x1, [x_shape[0] * x_shape[0], x_shape[1]])
    x2_2 = tf.reshape(x2, [x_shape[0] * x_shape[0], x_shape[1]])
    cond = tf.reduce_all(tf.equal(x1_2, x2_2), axis=1)
    cond = tf.reshape(cond, [x_shape[0], x_shape[0]])  # reshaping cond to match x1_2 & x2_2
    cond_shape = tf.shape(cond)
    cond_cast = tf.cast(cond, tf.int32)  # convertin condition boolean to int
    cond_zeros = tf.zeros(cond_shape, tf.int32)  # replicating condition tensor into all 0's

    # CREATING RANGE TENSOR
    r = tf.range(x_shape[0])
    r = tf.add(tf.tile(r, [x_shape[0]]), 1)
    r = tf.reshape(r, [x_shape[0], x_shape[0]])

    # converting TRUE=1 FALSE=MAX(index)+1 (which is invalid by default) so when we take min it wont get selected & in end we will only take values <max(indx).
    f1 = tf.multiply(tf.ones(cond_shape, tf.int32), x_shape[0] + 1)
    f2 = tf.ones(cond_shape, tf.int32)
    cond_cast2 = tf.where(tf.equal(cond_cast, cond_zeros), f1, f2)  # if false make it max_index+1 else keep it 1

    # multiply range with new int boolean mask
    r_cond_mul = tf.multiply(r, cond_cast2)
    r_cond_mul2 = tf.reduce_min(r_cond_mul, axis=1)
    r_cond_mul3, unique_idx = tf.unique(r_cond_mul2)
    r_cond_mul4 = tf.subtract(r_cond_mul3, 1)

    # get actual values from unique indexes
    op = tf.gather(x, r_cond_mul4)

    return (op.shape[0])
class TargetThroughput(tf.keras.metrics.Metric):
    def __init__(self, name='expected_throughput', logit=True, bit_string = True, **kwargs):
        super(TargetThroughput, self).__init__(name=name, **kwargs)
        self.max_throughput = self.add_weight(name='tp', initializer='zeros')
        self.expected_throughput = self.add_weight(name='tp2', initializer='zeros')
        self.count = self.add_weight(name='tp3', initializer='zeros')
        self.a = tf.math.log(tf.cast(2, dtype=tf.float32))
        self.logit = logit
        self.bit_string = bit_string

    def update_state(self, y_true, y_pred, x):
        if self.bit_string:
            mod_x = floatbits_to_float(x)
        else:
            mod_x = x
        c_max = tf.gather(mod_x, y_true, axis=1, batch_dims=1)
        i_max = tf.math.log(1 - 1.5*tf.math.log(1 - c_max))/self.a
        c_picked = tf.argmax(y_pred, axis=1)
        c_picked = tf.gather(mod_x, c_picked, axis=1, batch_dims=1)
        i_expected = tf.math.log(1 - 1.5*tf.math.log(1 - c_picked))/self.a
        self.max_throughput.assign_add(tf.math.reduce_sum(i_max, axis=0))
        self.expected_throughput.assign_add(tf.math.reduce_sum(i_expected, axis=0))
        self.count.assign_add(y_true.shape[0])

    def result(self):
        return self.max_throughput/self.count, self.expected_throughput/self.count
# ============================  Loss fn  ============================
def Encoding_variance():
    def encoding_variance(encode):
        loss = -tf.reduce_sum(tf.math.reduce_variance(encode, axis=0))
        return loss
    return encoding_variance
def Encoding_distance():
    def encoding_distance(encode):
        loss = 0
        for i in range(0, encode.shape[0]-1):
            loss = loss + tf.norm((encode[i], encode[i+1]))
        return -loss/encode.shape[0]
    return encoding_distance
def Loss_NN_encoding_diversity():
    model_path = "trained_models/Encoding Diversity/MLP_loss_function_2.h5"
    loss_model = tf.keras.models.load_model(model_path)
    for item in loss_model.layers:
        item.trainable = False
    return loss_model
def Regularization_loss():
    def regulariztion_loss(y_pred):
        loss = -tf.reduce_sum(tf.square(y_pred))/(y_pred.shape[0] * y_pred.shape[1])
        return loss
    return regulariztion_loss
def Throughout_diff_Loss():
    def through_put_loss(y_true, y_pred, x=None):
        a = tf.math.log(tf.cast(2, dtype=tf.float32))
        c_max = tf.gather(x, y_true, axis=1, batch_dims=1)
        i_max = tf.math.log(1 - 1.5 * tf.math.log(1 - c_max)) / a
        weighted_c = tf.math.multiply(Softmax()(y_pred), x)
        i_expected = tf.math.reduce_sum(tf.math.log(1 - 1.5 * tf.math.log(1 - weighted_c)) / a, axis=1)
        return tf.square(tf.math.reduce_sum(i_max, axis=0) - tf.math.reduce_sum(i_expected, axis=0))
    return through_put_loss
def ThroughputLoss():
    def through_put_loss(y_true, y_pred, x=None):
        a = tf.math.log(tf.cast(2, dtype=tf.float32))
        max_x = tf.argmax(y_pred, axis=1)
        c_chosen = tf.gather(x, max_x, axis=1, batch_dims=1)
        i_expected = tf.math.log(1 - 1.5 * tf.math.log(1 - c_chosen)) / a,
        cost = -tf.math.reduce_sum(i_expected, axis=0)
        return cost
    return through_put_loss
def ExpectedThroughputLoss():
    def through_put_loss(y_pred, x=None):
        # apply softmax to get distribution
        a = tf.math.log(tf.cast(2, dtype=tf.float32))
        weighted_c = tf.math.multiply(Softmax()(y_pred), x)
        i_expected = tf.math.reduce_sum(tf.math.log(1 - 1.5 * tf.math.log(1 - weighted_c)) / a, axis=1)
        cost = -tf.math.reduce_sum(i_expected, axis=0)
        return cost
    return through_put_loss
def Mix_loss():
    def mixloss(y_true, y_pred, x=None):
        a = tf.math.log(tf.cast(2, dtype=tf.float32))
        c_max = tf.gather(x, y_true, axis=1, batch_dims=1)
        i_max = tf.math.log(1 - 1.5 * tf.math.log(1 - c_max)) / a
        weighted_c = tf.math.multiply(Softmax()(y_pred), x)
        i_expected = tf.math.reduce_sum(tf.math.log(1 - 1.5 * tf.math.log(1 - weighted_c)) / a, axis=1)
        loss1 = tf.square(tf.math.reduce_sum(i_max, axis=0) - tf.math.reduce_sum(i_expected, axis=0))
        loss2 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred)
        return loss2 + loss1/100
    return mixloss
def user_constraint(pred_i, K, M):
    unflattened_X = tf.reshape(pred_i, (pred_i.shape[0], K, M))
    loss = tf.reduce_mean(tf.square(tf.maximum(tf.reduce_sum(unflattened_X, axis=2), 1.0)-1.0))
    return loss
def non_double_count_loss(pred_i, K, M):
    unflattened_X = tf.reshape(pred_i, (pred_i.shape[0], K, M))
    loss = tf.reduce_mean(tf.square(tf.multiply(unflattened_X, 1.0-unflattened_X)))
    return loss
def Negative_shove():
    def negative_shove(y_pred, x=None):
        values, indices = tf.nn.top_k(y_pred, 2)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(values[:,1], y_pred)
        return -loss
    return negative_shove
def CE_with_distribution():
    loss_fn = tf.keras.losses.CategoricalCrossentropy
    def loss(prediction, label):
        return loss_fn(prediction, label)
def Binarization_regularization(ranking=False):
    def regularization(y_pred):
        loss = -tf.square(2 * (y_pred - 0.5))
        loss = tf.reduce_mean(loss, axis = 1)
        return loss
    return regularization
def OutPut_Limit(N_rf):
    def regularization(y_pred):
        loss = tf.reduce_sum(y_pred, axis=1)
        loss = tf.maximum(-(loss-tf.constant(N_rf, dtype=tf.float32)), (loss-tf.constant(N_rf, dtype=tf.float32)))
        return loss
    return regularization
def OutPut_Limit_onesided(N_rf):
    def regularization(y_pred):
        loss = tf.reduce_sum(y_pred, axis=1)
        loss = tf.maximum(0, (loss-tf.constant(N_rf, dtype=tf.float32)))
        return loss
    return regularization
def Output_Per_Receiver_Control(K, M, ranking=False):
    def regularization(y_pred):
        loss = 0
        y_pred_list = tf.split(y_pred, num_or_size_splits=K, axis=1)
        for per_device_y_pred in y_pred_list:
            loss += tf.square(tf.reduce_sum(per_device_y_pred, axis=1) - 1)
        return loss
    return regularization
def Total_activation_limit_soft(K, M, ranking=False, N_rf = 3):
    def regularization(y_pred):
        y_pred_mod = y_pred
        if ranking:
            y_pred_mod = y_pred[:K*M]
        sum = tf.reduce_sum(y_pred_mod, axis=1)
        loss = tf.square(sum - N_rf)
        # print(sum.shape)
        return loss
    return regularization
def Total_activation_limit_hard(K, M, ranking=False, N_rf = 3):
    def regularization(y_pred):
        y_pred = y_pred + tf.stop_gradient(binary_activation(y_pred) - y_pred)
        y_pred_mod = y_pred
        if ranking:
            y_pred_mod = y_pred[:K*M]
        sum = tf.reduce_sum(y_pred_mod, axis=1)
        loss = tf.reduce_mean(sum)
        return loss
    return regularization
def Harden_scheduling(k=3, K=0, M=0, sigma2=0):
    def masking(y_pred):
        # assumes the input shape is (batch, k*M) for y_pred,
        # and the shape for G is (batch, K, N)
        # generate mask to mask out points that are not in top k vvvvv
        base_mask = np.zeros((y_pred.shape))
        values, index = tf.math.top_k(y_pred, k=k)
        for i in range(0, y_pred.shape[0]):
            base_mask[i, index[i]] = 1
        return tf.constant(base_mask, dtype=tf.float32)
    return masking
def Harden_scheduling_user_constrained(N_rf=3, K=0, M=0, sigma2=0, default_val = 0.0):
    def masking(y_pred):
        # assumes the input shape is (batch, k*M) for y_pred,
        # and the shape for G is (batch, K, N)
        # generate mask to mask out points that are not in top k vvvvv
        base_mask = np.ones((y_pred.shape)) * default_val
        try:
            y_pred_np = y_pred.numpy()
        except:
            y_pred_np = y_pred[:]
        y_pred_copy = np.zeros((y_pred.shape))
        for k in range(K):
            y_pred_argmax = np.argmax(y_pred_np[:, k*M:(k+1)*M], axis=1)
            for n in range(0, y_pred_np.shape[0]):
                y_pred_copy[n, k*M+y_pred_argmax[n]] = y_pred_np[n, k*M+y_pred_argmax[n]]
        values, index = tf.math.top_k(y_pred_copy, k=N_rf)
        for i in range(0, y_pred.shape[0]):
            base_mask[i, index[i]] = 1.0
        return tf.constant(base_mask, dtype=tf.float32)
    return masking
def Harden_scheduling_neg(k=3, K=0, M=0, sigma2=0):
    def masking(y_pred):
        # assumes the input shape is (batch, k*M) for y_pred,
        # and the shape for G is (batch, K, N)
        # generate mask to mask out points that are not in top k vvvvv
        base_mask = np.ones((y_pred.shape))*(1.0/K/M)
        values, index = tf.math.top_k(y_pred, k=k)
        for i in range(0, y_pred.shape[0]):
            base_mask[i, index[i]] = 1
        return tf.constant(base_mask, dtype=tf.float32)
    return masking
def Masking_with_learned_weights(K, M, sigma2, k=3):
    stretch_matrix = np.zeros((M * K, K))
    for i in range(0, K):
        for j in range(0, M):
            stretch_matrix[i * M + j, i] = 1
    stretch_matrix = tf.constant(stretch_matrix, tf.float32)
    def masking(y_pred):
        # assumes the input shape is (batch, k*M) for y_pred,
        # and the shape for G is (batch, K, N)
        y_pred_val = y_pred[:, 0:K*M]
        y_pred_rank = y_pred[:, K*M:]
        y_pred_rank = tf.reshape(y_pred_rank, (y_pred_rank.shape[0], y_pred_rank.shape[1], 1))
        tiled_stretch_matrix = tf.tile(tf.expand_dims(stretch_matrix, 0), [y_pred_rank.shape[0], 1, 1])
        stretched_rank_matrix = tf.matmul(tiled_stretch_matrix, y_pred_rank)
        stretched_rank_matrix = tf.reshape(stretched_rank_matrix, (stretched_rank_matrix.shape[0], stretched_rank_matrix.shape[1]))
        y_pred_val = tf.multiply(stretched_rank_matrix, y_pred_val)
        # generate mask to mask out points that are not in top k vvvvv
        values, index = tf.math.top_k(y_pred_val, k=k)
        base_mask = np.zeros((y_pred_val.shape))
        for i in range(0, y_pred_val.shape[0]):
            base_mask[i, index[i]] = 1
        y_pred_val = tf.multiply(y_pred_val, base_mask)
        normalization_mask = tf.where(y_pred_val==0, 1, y_pred_val)
        y_pred_val = y_pred_val + tf.stop_gradient(y_pred_val/normalization_mask - y_pred_val)
        return y_pred_val
    return masking
def Masking_with_learned_weights_soft(K, M, sigma2, k=3):
    stretch_matrix = np.zeros((M * K, K))
    for i in range(0, K):
        for j in range(0, M):
            stretch_matrix[i * M + j, i] = 1
    stretch_matrix = tf.constant(stretch_matrix, tf.float32)
    def masking(y_pred):
        # assumes the input shape is (batch, k*M) for y_pred,
        # and the shape for G is (batch, K, N)
        y_pred_val = y_pred[:, 0:K*M]
        y_pred_rank = y_pred[:, K*M:]
        y_pred_rank = tf.reshape(y_pred_rank, (y_pred_rank.shape[0], y_pred_rank.shape[1], 1))
        # print(y_pred_rank[0])
        tiled_stretch_matrix = tf.tile(tf.expand_dims(stretch_matrix, 0), [y_pred_rank.shape[0], 1, 1])
        stretched_rank_matrix = tf.matmul(tiled_stretch_matrix, y_pred_rank)
        stretched_rank_matrix = tf.reshape(stretched_rank_matrix, (stretched_rank_matrix.shape[0], stretched_rank_matrix.shape[1]))
        y_pred_val = tf.multiply(stretched_rank_matrix, y_pred_val)
        # generate mask to mask out points that are not in top k vvvvv
        return y_pred_val
    return masking
def Masking_with_ranking_prob(K, M, sigma2, k=3):
    stretch_matrix = np.zeros((M * K, K))
    for i in range(0, K):
        for j in range(0, M):
            stretch_matrix[i * M + j, i] = 1
    stretch_matrix = tf.constant(stretch_matrix, tf.float32)
    def masking(y_pred):
        # assumes the input shape is (batch, k*M) for y_pred,
        # and the shape for G is (batch, K, N)
        y_pred_val = y_pred
        # generate mask to mask out points that are not in top k vvvvv
        values, index = tf.math.top_k(y_pred_val, k=k)
        print(values)
        base_mask = np.zeros((y_pred_val.shape))
        for i in range(0, y_pred_val.shape[0]):
            base_mask[i, index[i]] = 1
        y_pred_val = tf.multiply(y_pred_val, base_mask)
        normalization_mask = tf.where(y_pred_val == 0, 1, y_pred_val)
        y_pred_val = y_pred_val + tf.stop_gradient(y_pred_val / normalization_mask - y_pred_val)
        return y_pred_val
    return masking
def TEMP_Pairwise_Cross_Entropy_loss(K, M, k):
    def loss_fn(y_pred):
        # assumes the input shape is (batch, k*M) for y_pred,
        # and the shape for G is (batch, K, N)
        y_pred_rank = y_pred[:, K * M:]
        y_pred_rank = tf.reshape(y_pred_rank, (y_pred_rank.shape[0], k, K))
        loss = tf.reduce_sum(y_pred_rank[:, 0] * tf.math.log(y_pred_rank[:, 1]))
        loss += tf.reduce_sum(y_pred_rank[:, 1] * tf.math.log(y_pred_rank[:, 2]))
        loss += tf.reduce_sum(y_pred_rank[:, 0] * tf.math.log(y_pred_rank[:, 2]))
        loss = loss
        return loss
    return loss_fn
def  _plusp5(K, M, sigma2):
    # sigma2 here is the variance of the noise
    log_2 = tf.math.log(tf.constant(2.0, dtype=tf.float32))
    def sum_rate_utility(y_pred, G, display=False):
        # assumes the input shape is (batch, k*N) for y_pred,
        # and the shape for G is (batch, K, M)
        G = tf.square(tf.abs(G))
        unflattened_X = tf.reshape(y_pred, (y_pred.shape[0], K, M))
        unflattened_X = tf.transpose(unflattened_X, perm=[0, 2, 1])
        denominator = tf.matmul(G, unflattened_X)
        if display:
            plt.imshow(denominator[0])
            plt.show(block=False)
            plt.pause(0.0001)
            plt.close()
        numerator = tf.multiply(denominator, tf.eye(K))
        denominator = tf.reduce_sum(denominator-numerator, axis=2) + sigma2
        numerator = tf.matmul(numerator, tf.ones((K, 1)))
        numerator = tf.reshape(numerator, (numerator.shape[0], numerator.shape[1]))
        utility = tf.math.log(numerator/denominator+0.5)/log_2
        utility = tf.reduce_sum(utility, axis=1)
        return -utility
    return sum_rate_utility
def sinkhorn(X, n):
    X = tf.exp(X)
    for i in range(n):
        X = tf.divide(X, tf.reduce_sum(X, axis=1, keepdims=True))
    for i in range(n):
        X = tf.divide(X, tf.reduce_sum(X, axis=2, keepdims=True))
    decision = tf.reduce_sum(X, axis=2)
    return decision
def Sum_rate_utility_WeiCui(K, M, sigma2):
    # sigma2 here is the variance of the noise
    log_2 = tf.math.log(tf.constant(2.0, dtype=tf.float32))
    def sum_rate_utility(y_pred, G, display=False):
        # assumes the input shape is (batch, k*N) for y_pred,
        # and the shape for G is (batch, K, M)
        G = tf.square(tf.abs(G))
        unflattened_X = tf.reshape(y_pred, (y_pred.shape[0], K, M))
        unflattened_X = tf.transpose(unflattened_X, perm=[0, 2, 1])
        denominator = tf.matmul(G, unflattened_X)
        if display:
            plt.imshow(denominator[0])
            plt.show(block=False)
            plt.pause(0.0001)
            plt.close()
        numerator = tf.multiply(denominator, tf.eye(K))
        denominator = tf.reduce_sum(denominator-numerator, axis=2) + sigma2
        numerator = tf.matmul(numerator, tf.ones((K, 1)))
        numerator = tf.reshape(numerator, (numerator.shape[0], numerator.shape[1]))
        utility = tf.math.log(numerator/denominator + 1)/log_2
        utility = tf.reduce_sum(utility, axis=1)
        return -utility
    return sum_rate_utility
def Sum_rate_utility_WeiCui_stable(K, M, sigma2, constant=0.1):
    # sigma2 here is the variance of the noise
    log_2 = tf.math.log(tf.constant(2.0, dtype=tf.float32))
    def sum_rate_utility(y_pred, G, display=False):
        # assumes the input shape is (batch, k*N) for y_pred,
        # and the shape for G is (batch, K, M)
        G = tf.square(tf.abs(G))
        unflattened_X = tf.reshape(y_pred, (y_pred.shape[0], K, M))
        unflattened_X = tf.transpose(unflattened_X, perm=[0, 2, 1])
        denominator = tf.matmul(G, unflattened_X)
        numerator = tf.multiply(denominator, tf.eye(K))
        denominator = tf.reduce_sum(denominator-numerator, axis=2) + sigma2
        numerator = tf.matmul(numerator, tf.ones((K, 1)))
        numerator = tf.reshape(numerator, (numerator.shape[0], numerator.shape[1])) + constant
        utility = tf.math.log(numerator/denominator + 1)/log_2
        utility = tf.reduce_sum(utility, axis=1)
        return -utility
    return sum_rate_utility
def Sum_rate_utility_WeiCui_seperate_user_stable(K, M, sigma2, constant=0.1):
    # sigma2 here is the variance of the noise
    log_2 = tf.math.log(tf.constant(2.0, dtype=tf.float32))
    def sum_rate_utility(y_pred, G, display=False):
        # assumes the input shape is (batch, k*N) for y_pred,
        # and the shape for G is (batch, K, M)
        G = tf.square(tf.abs(G))
        unflattened_X = tf.reshape(y_pred, (y_pred.shape[0], K, M))
        unflattened_X = tf.transpose(unflattened_X, perm=[0, 2, 1])
        denominator = tf.matmul(G, unflattened_X)
        if display:
            plt.imshow(denominator[0])
            plt.show(block=False)
            plt.pause(0.0001)
            plt.close()
        numerator = tf.multiply(denominator, tf.eye(K))
        denominator = tf.reduce_sum(denominator-numerator, axis=2) + sigma2
        numerator = tf.matmul(numerator, tf.ones((K, 1)))
        numerator = tf.reshape(numerator, (numerator.shape[0], numerator.shape[1])) + constant
        utility = tf.math.log(numerator/denominator + 1)/log_2
        return utility
    return sum_rate_utility
def Sum_rate_utility_WeiCui_seperate_user(K, M, sigma2):
    # sigma2 here is the variance of the noise
    log_2 = tf.math.log(tf.constant(2.0, dtype=tf.float32))
    def sum_rate_utility(y_pred, G, display=False):
        # assumes the input shape is (batch, k*N) for y_pred,
        # and the shape for G is (batch, K, M)
        G = tf.square(tf.abs(G))
        unflattened_X = tf.reshape(y_pred, (y_pred.shape[0], K, M))
        unflattened_X = tf.transpose(unflattened_X, perm=[0, 2, 1])
        denominator = tf.matmul(G, unflattened_X)
        if display:
            plt.imshow(denominator[0])
            plt.show(block=False)
            plt.pause(0.0001)
            plt.close()
        numerator = tf.multiply(denominator, tf.eye(K))
        denominator = tf.reduce_sum(denominator-numerator, axis=2) + sigma2
        numerator = tf.matmul(numerator, tf.ones((K, 1)))
        numerator = tf.reshape(numerator, (numerator.shape[0], numerator.shape[1]))
        utility = tf.math.log(numerator/denominator + 1)/log_2
        return utility
    return sum_rate_utility
def Sum_rate_interference(K, M, sigma2):
    # sigma2 here is the variance of the noise
    log_2 = tf.math.log(tf.constant(2.0, dtype=tf.float32))
    def sum_rate_utility(y_pred, G, display=False):
        # assumes the input shape is (batch, k*N) for y_pred,
        # and the shape for G is (batch, K, M)
        G = tf.square(tf.abs(G))
        unflattened_X = tf.reshape(y_pred, (y_pred.shape[0], K, M))
        unflattened_X = tf.transpose(unflattened_X, perm=[0, 2, 1])
        denominator = tf.matmul(G, unflattened_X)
        if display:
            plt.imshow(denominator[0])
            plt.show(block=False)
            plt.pause(0.0001)
            plt.close()
        numerator = tf.multiply(denominator, tf.eye(K))
        denominator = tf.reduce_sum(denominator-numerator, axis=2) + sigma2
        utility = tf.math.log(1 / denominator) / log_2
        utility = tf.reduce_mean(utility, axis=1)
        return utility
    return sum_rate_utility
def Sum_rate_utility(K, M, sigma2):
    # sigma2 here is the variance of the noise
    def sum_rate_utility(y_pred, G):
        # assumes the input shape is (batch, k*N) for y_pred,
        # and the shape for G is (batch, K, N)
        g_flatten = tf.reshape(G, (G.shape[0], K*M))
        g_flatten = tf.square(tf.abs(g_flatten))
        sum_vector = tf.reduce_sum(tf.multiply(y_pred, g_flatten) + sigma2, axis=1)
        numerator = tf.multiply(g_flatten, y_pred)
        denominator = tf.subtract(tf.reshape(sum_vector, (sum_vector.shape[0], 1)), numerator) + tf.constant(sigma2)
        utility = tf.math.log(numerator/denominator+0.00000001 + 1)/tf.math.log(10)
        utility = tf.reduce_sum(utility, axis=1)
        return -utility
    return sum_rate_utility
def Sum_rate_utility_hard(K, M, sigma2):
    log_2 = tf.math.log(tf.constant(2.0, dtype=tf.float32))
    def sum_rate_utility(y_pred, mask, G, display=False):
        # assumes the input shape is (batch, k*N) for y_pred,
        # and the shape for G is (batch, K, M)
        G = tf.square(tf.abs(G))
        unflattened_X = tf.reshape(y_pred, (y_pred.shape[0], K, M))
        unflattened_X = tf.transpose(unflattened_X, perm=[0, 2, 1])
        denominator = tf.matmul(G, unflattened_X)
        mask = tf.reduce_sum(tf.keras.layers.Reshape((K, M))(mask), axis=2)
        numerator = tf.multiply(denominator, tf.eye(K))
        denominator = tf.reduce_sum(denominator - numerator, axis=2) + sigma2
        numerator = tf.matmul(numerator, tf.ones((K, 1)))
        numerator = tf.reshape(numerator, (numerator.shape[0], numerator.shape[1]))
        numerator = tf.multiply(mask, numerator)
        utility = tf.math.log(numerator / denominator + 1) / log_2
        utility = tf.reduce_sum(utility, axis=1)
        return -utility

    return sum_rate_utility
def Sum_rate_matrix_CE(K, M, sigma2):
    # sigma2 here is the variance of the noise
    log_2 = tf.math.log(tf.constant(2.0, dtype=tf.float32))
    matrix_goal_template = tf.reshape(tf.eye(K), (K*K, ))
    def sum_rate_utility(y_pred, G, display=False):
        G = tf.square(tf.abs(G))
        unflattened_X = tf.reshape(y_pred, (y_pred.shape[0], K, M))
        unflattened_X = tf.transpose(unflattened_X, perm=[0, 2, 1])
        matrix = tf.matmul(G, unflattened_X)
        matrix = tf.keras.layers.Softmax(axis=1)(matrix)
        matrix = tf.keras.layers.Reshape((K*K,))(matrix)
        # matrix_goal_current = tf.reduce_max(matrix, axis=1)
        matrix_goal_template_tiled = tf.tile(tf.expand_dims(matrix_goal_template, 0), (y_pred.shape[0], 1))
        # matrix_goal_current = tf.multiply(matrix_goal_current, matrix_goal_template_tiled)
        utility = tf.losses.CategoricalCrossentropy(from_logits=False)(matrix_goal_template_tiled, matrix)
        # utility = tf.reduce_sum(utility, axis=1)
        return utility
    return sum_rate_utility
def Sum_rate_utility_bad_Talor(K, M, sigma2):
    # sigma2 here is the variance of the noise
    log_2 = tf.math.log(tf.constant(2.0, dtype=tf.float32))
    # stretch_matrix = np.zeros((K, K*M))
    # for i in range(0, K):
    #     for j in range(0, M):
    #         stretch_matrix[i, i * M + j] = 1
    # stretch_matrix = tf.constant(stretch_matrix, tf.float32)
    def sum_rate_utility(y_pred, G, display=False):
        # assumes the input shape is (batch, k*N) for y_pred,
        # and the shape for G is (batch, K, M)
        G = tf.square(tf.abs(G))
        unflattened_X = tf.reshape(y_pred, (y_pred.shape[0], K, M))
        unflattened_X = tf.transpose(unflattened_X, perm=[0, 2, 1])
        denominator = tf.matmul(G, unflattened_X)
        numerator = tf.multiply(denominator, tf.eye(K))
        denominator = tf.reduce_sum(denominator-numerator, axis=2) + sigma2
        numerator = tf.matmul(numerator, tf.ones((K, 1)))
        numerator = tf.reshape(numerator, (numerator.shape[0], numerator.shape[1]))
        utility = tf.math.log(numerator+1.0) - (denominator - 1.0 - 0.5*tf.square(denominator - 1.0))
        utility = tf.reduce_sum(utility, axis=1)
        return -utility
    return sum_rate_utility
def Sum_rate_difference(K, M, sigma2):
    # sigma2 here is the variance of the noise
    log_2 = tf.math.log(tf.constant(2.0, dtype=tf.float32))
    # stretch_matrix = np.zeros((K, K*M))
    # for i in range(0, K):
    #     for j in range(0, M):
    #         stretch_matrix[i, i * M + j] = 1
    # stretch_matrix = tf.constant(stretch_matrix, tf.float32)
    def sum_rate_utility(y_pred, G, display=False):
        # assumes the input shape is (batch, k*N) for y_pred,
        # and the shape for G is (batch, K, M)
        G = tf.square(tf.abs(G))
        unflattened_X = tf.reshape(y_pred, (y_pred.shape[0], K, M))
        unflattened_X = tf.transpose(unflattened_X, perm=[0, 2, 1])
        denominator = tf.matmul(G, unflattened_X)
        numerator = tf.multiply(denominator, tf.eye(K))
        denominator = tf.reduce_sum(denominator-numerator, axis=2) + sigma2
        numerator = tf.matmul(numerator, tf.ones((K, 1)))
        numerator = tf.reshape(numerator, (numerator.shape[0], numerator.shape[1]))
        utility = numerator - denominator
        utility = tf.reduce_sum(utility, axis=1)
        return -utility
    return sum_rate_utility
def Sum_rate_utility_RANKING_hard(K, M, sigma2, k, display_all=False):
    sr = Sum_rate_utility_WeiCui(K, M, sigma2)
    def cal_sum_rate(y_pred, G):
        loss = sr(Harden_scheduling(1)(y_pred[:, :, 0]), G)
        if display_all:
            print(tf.reduce_mean(loss))
        for i in range(1, int(k)):
            rate = sr(Harden_scheduling(1+i)(y_pred[:, :, i]), G)
            loss = loss + rate
            if display_all:
                print(tf.reduce_mean(rate))
        return loss
    return cal_sum_rate
def Sum_rate_utility_RANKING(K, M, sigma2, k, display_all=False):
    sr = Sum_rate_utility_WeiCui(K, M, sigma2)
    def cal_sum_rate(y_pred, G):
        loss = sr(y_pred[:, :, 0], G)
        if display_all:
            print(tf.reduce_mean(loss))
        for i in range(1, int(k)):
            rate = sr(y_pred[:, :, i], G)
            loss = loss + rate
            if display_all:
                print(tf.reduce_mean(rate))
        return loss
    return cal_sum_rate
def Verti_sum_utility_RANKING(K, M, sigma2, k):
    sr = Sum_rate_utility_WeiCui_wrong_axis(K, M, sigma2)
    def cal_sum_rate(y_pred, G):
        loss = sr(y_pred[:, :, 0], G)
        for i in range(1, int(k)):
            loss = loss + sr(y_pred[:, :, i], G)
        return loss
    return cal_sum_rate
def ensumble_output(G, model, k, loss_fn):
    output = tf.keras.layers.Reshape((G.shape[1]*G.shape[2], 1))(model.predict(G)[:, -1, :])
    sr = np.zeros((G.shape[0], k))
    for i in range(1, k):
        inputs = tf.random.shuffle(G)
        output_i = tf.keras.layers.Reshape((G.shape[1] * G.shape[2], 1))(model.predict(inputs)[:, -1, :])
        output = tf.concat((output, output_i), axis=2)
        sr[:, i] = loss_fn(output_i[:, :, 0], inputs)
        print("round ",str(i), "done!")
    sr_final = tf.reduce_min(sr, axis=1)
    print(tf.reduce_mean(sr_final))
    max_indices = tf.argmax(sr, axis=1)
    print(max_indices.shape)
    output = tf.gather(output, max_indices, axis=2, batch_dims=1)
    print(output.shape)
    return output
def nrf2expected_loss(N_rf, M, K, sigma):
    def loss_fn(G, y_pred):
        Gp1 = tf.square(tf.abs(G)) + 1.0
        M_Gp1 = tf.keras.layers.Reshape((K*M, 1))(Gp1)
        M_y_pred = tf.keras.layers.Reshape((K*M, 1))(y_pred)
        tim = tf.multiply(tf.math.log(tf.matmul(M_Gp1, tf.transpose(M_Gp1, (0, 2, 1)))), tf.matmul(M_y_pred, tf.transpose(M_y_pred, (0, 2, 1))))
        tim = tf.reduce_sum(tf.reduce_sum(tim, axis=1), axis=1)
        loss = tf.reduce_mean(tim)
        return -loss
    return loss_fn
def mutex_loss(N_rf, M, K, N):
    def loss_fn(out):
        sm_out = tf.keras.layers.Softmax(axis=1)(out)
        sub1 = tf.tile(tf.expand_dims(1 - sm_out, axis=3), [1,1,1,N_rf])
        one = tf.tile(tf.expand_dims(tf.expand_dims(tf.ones((N_rf, N_rf)), axis=0), axis=1), [N, M*K, 1, 1])
        one = tf.multiply(one, sub1)
        sub1 = sub1 - one
        one = tf.reduce_sum(one, axis=2)
        loss = tf.multiply(one, tf.reduce_prod(sub1, axis=3))
        loss = tf.reduce_mean(loss)
        return -loss
    return loss_fn

def Sum_rate_utility_WeiCui_all_link_streaming(K, M, sigma2):
    # sigma2 here is the variance of the noise
    log_2 = tf.math.log(tf.constant(2.0, dtype=tf.float32))
    # stretch_matrix = np.zeros((K, K*M))
    # for i in range(0, K):
    #     for j in range(0, M):
    #         stretch_matrix[i, i * M + j] = 1
    # stretch_matrix = tf.constant(stretch_matrix, tf.float32)
    def sum_rate_utility(y_pred, G, display=False):
        # assumes the input shape is (batch, k*N) for y_pred,
        # and the shape for G is (batch, K, M)
        G = tf.square(tf.abs(G))
        unflattened_X = tf.reshape(y_pred, (y_pred.shape[0], K, M))
        unflattened_X = tf.transpose(unflattened_X, perm=[0, 2, 1])
        denominator = tf.matmul(G, unflattened_X)
        if display:
            plt.imshow(denominator[0])
            plt.show(block=False)
            plt.pause(0.0001)
            plt.close()
        numerator = tf.multiply(denominator, tf.eye(K))
        denominator = tf.reduce_sum(denominator-numerator, axis=2) + sigma2
        numerator = tf.matmul(numerator, tf.ones((K, 1)))
        numerator = tf.reshape(numerator, (numerator.shape[0], numerator.shape[1]))
        utility = 5*tf.math.log((numerator + 0.1)/denominator + 1)/log_2
        utility = tf.reduce_sum(utility, axis=1)
        return -utility
    return sum_rate_utility
def Sum_rate_utility_WeiCui_wrong_axis(K, M, sigma2):
    # sigma2 here is the variance of the noise
    log_2 = tf.math.log(tf.constant(2.0, dtype=tf.float32))
    # stretch_matrix = np.zeros((K, K*M))
    # for i in range(0, K):
    #     for j in range(0, M):
    #         stretch_matrix[i, i * M + j] = 1
    # stretch_matrix = tf.constant(stretch_matrix, tf.float32)
    def sum_rate_utility(y_pred, G):
        # assumes the input shape is (batch, k*N) for y_pred,
        # and the shape for G is (batch, K, M)
        G = tf.square(tf.abs(G))
        unflattened_X = tf.reshape(y_pred, (y_pred.shape[0], K, M))
        unflattened_X = tf.transpose(unflattened_X, perm=[0, 2, 1])
        denominator = tf.matmul(G, unflattened_X)
        numerator = tf.multiply(denominator, tf.eye(K))
        denominator = tf.reduce_sum(denominator-numerator, axis=1) + sigma2
        numerator = tf.matmul(numerator, tf.ones((K, 1)))
        numerator = tf.reshape(numerator, (numerator.shape[0], numerator.shape[1]))
        utility = tf.math.log((numerator)/denominator + 1)/log_2
        # utility = numerator/denominator
        utility = tf.reduce_sum(utility, axis=1)
        return -utility
    return sum_rate_utility
def Sum_rate_utility_WeiCui_wrong_axis_with_constant(K, M, sigma2):
    # sigma2 here is the variance of the noise
    log_2 = tf.math.log(tf.constant(2.0, dtype=tf.float32))
    # stretch_matrix = np.zeros((K, K*M))
    # for i in range(0, K):
    #     for j in range(0, M):
    #         stretch_matrix[i, i * M + j] = 1
    # stretch_matrix = tf.constant(stretch_matrix, tf.float32)
    def sum_rate_utility(y_pred, G):
        # assumes the input shape is (batch, k*N) for y_pred,
        # and the shape for G is (batch, K, M)
        G = tf.square(tf.abs(G))
        unflattened_X = tf.reshape(y_pred, (y_pred.shape[0], K, M))
        unflattened_X = tf.transpose(unflattened_X, perm=[0, 2, 1])
        denominator = tf.matmul(G, unflattened_X)
        numerator = tf.multiply(denominator, tf.eye(K))
        denominator = tf.reduce_sum(denominator-numerator, axis=1) + sigma2
        numerator = tf.matmul(numerator, tf.ones((K, 1)))
        numerator = tf.reshape(numerator, (numerator.shape[0], numerator.shape[1]))
        utility = 5*tf.math.log((numerator+0.1)/denominator + 1)/log_2
        utility = tf.reduce_sum(utility, axis=1)
        return -utility
    return sum_rate_utility
def Sum_rate_utility_top_k_with_mask_from_learned_weights(K, M, sigma2, k=3):
    # sigma2 here is the variance of the noise
    log_2 = tf.math.log(tf.constant(2.0, dtype=tf.float32))
    stretch_matrix = np.zeros((M*K, K))
    for i in range(0, K):
        for j in range(0, M):
            stretch_matrix[i*M + j, i] = 1
    stretch_matrix = tf.constant(stretch_matrix, tf.float32)
    def sum_rate_utility(y_pred, G):
        # assumes the input shape is (batch, k*M) for y_pred,
        # and the shape for G is (batch, K, N)
        y_pred_val = y_pred[:, 0:K*M]
        y_pred_rank = y_pred[:, K*M:]
        y_pred_rank = tf.reshape(y_pred_rank, (y_pred_rank.shape[0], y_pred_rank.shape[1], 1))
        tiled_stretch_matrix = tf.tile(tf.expand_dims(stretch_matrix, 0), [y_pred_rank.shape[0], 1, 1])
        stretched_rank_matrix = tf.matmul(tiled_stretch_matrix, y_pred_rank)
        stretched_rank_matrix = tf.reshape(stretched_rank_matrix, (stretched_rank_matrix.shape[0], stretched_rank_matrix.shape[1]))
        y_pred_val = tf.multiply(stretched_rank_matrix, y_pred_val)
        # generate mask to mask out points that are not in top k vvvvv
        values, index = tf.math.top_k(y_pred_val, k=k)
        base_mask = np.zeros((y_pred_val.shape))
        for i in range(0, y_pred_val.shape[0]):
            base_mask[i, index[i]] = 1
        y_pred_val = tf.multiply(y_pred_val, base_mask)
        normalization_mask = tf.where(y_pred_val==0, 1, y_pred_val)
        y_pred_val = y_pred_val + tf.stop_gradient(y_pred_val/normalization_mask - y_pred_val)
        # generate mask to mask out points that are not in top k ^^^^
        g_flatten = tf.reshape(G, (G.shape[0], K*M))
        g_flatten = tf.square(tf.abs(g_flatten))
        sum_vector = tf.reduce_sum(tf.multiply(y_pred_val, g_flatten), axis=1)
        numerator = tf.multiply(y_pred_val, g_flatten)
        denominator = 6*(tf.subtract(tf.reshape(sum_vector, (sum_vector.shape[0], 1)), numerator))
        # numerator = numerator + tf.stop_gradient(tf.round(numerator) - numerator)
        # denominator = denominator + tf.stop_gradient(tf.round(denominator) - denominator)
        utility = 5*tf.math.log(numerator/(denominator+0.00000001) + 1)/log_2
        utility = tf.reduce_sum(utility, axis=1)
        return -utility
    return sum_rate_utility
def Sum_rate_utility_top_k_with_mask_from_learned_ranking(K, M, sigma2, k=3):
    # sigma2 here is the variance of the noise
    log_2 = tf.math.log(tf.constant(2.0, dtype=tf.float32))
    def sum_rate_utility(y_pred, G):
        # assumes the input shape is (batch, k*M) for y_pred,
        # and the shape for G is (batch, K, N)
        y_pred_val = y_pred[:, 0:K*M]
        y_pred_rank = y_pred[:, K*M:]
        # generate mask to mask out points that are not in top k vvvvv
        values, base_index = tf.math.top_k(y_pred_rank, k=k)
        base_mask = np.zeros((y_pred_val.shape))
        index = np.array([], dtype=np.int64).reshape((y_pred_val.shape[0], 0))
        for i in range(0, M):
            index = np.concatenate((index, base_index*M+i), axis=1)
        index = np.sort(index, axis=1)
        for i in range(0, y_pred_val.shape[0]):
            base_mask[i, index[i]] = 1
        y_pred_val = tf.multiply(y_pred_val, base_mask)
        # generate mask to mask out points that are not in top k ^^^^
        g_flatten = tf.reshape(G, (G.shape[0], K*M))
        g_flatten = tf.square(tf.abs(g_flatten))
        sum_vector = tf.reduce_sum(tf.multiply(y_pred_val, g_flatten), axis=1)
        numerator = tf.multiply(y_pred_val, g_flatten)
        denominator = 6*(tf.subtract(tf.reshape(sum_vector, (sum_vector.shape[0], 1)), numerator))
        # numerator = numerator + tf.stop_gradient(tf.round(numerator) - numerator)
        # denominator = denominator + tf.stop_gradient(tf.round(denominator) - denominator)
        utility = 5*tf.math.log(numerator/(denominator+0.00000001) + 1)/log_2
        utility = tf.reduce_sum(utility, axis=1)
        return -utility
    return sum_rate_utility
def Sum_rate_utility_top_k_with_mask_from_ranking_prob(K, M, sigma2, k=3):
    # sigma2 here is the variance of the noise
    log_2 = tf.math.log(tf.constant(2.0, dtype=tf.float32))
    def sum_rate_utility(y_pred, G):
        # assumes the input shape is (batch, k*M) for y_pred,
        # and the shape for G is (batch, K, N)
        y_pred_val = y_pred
        # generate mask to mask out points that are not in top k vvvvv
        values, index = tf.math.top_k(y_pred_val, k=k)
        base_mask = np.zeros((y_pred_val.shape))
        for i in range(0, y_pred_val.shape[0]):
            base_mask[i, index[i]] = 1
        y_pred_val = tf.multiply(y_pred_val, base_mask)
        # generate mask to mask out points that are not in top k ^^^^
        g_flatten = tf.reshape(G, (G.shape[0], K*M))
        g_flatten = tf.square(tf.abs(g_flatten))
        sum_vector = tf.reduce_sum(tf.multiply(y_pred_val, g_flatten), axis=1)
        numerator = tf.multiply(y_pred_val, g_flatten)
        denominator = 6*(tf.subtract(tf.reshape(sum_vector, (sum_vector.shape[0], 1)), numerator))
        # numerator = numerator + tf.stop_gradient(tf.round(numerator) - numerator)
        # denominator = denominator + tf.stop_gradient(tf.round(denominator) - denominator)
        utility = 5*tf.math.log(numerator/(denominator+0.00000001) + 1)/log_2
        utility = tf.reduce_sum(utility, axis=1)
        return -utility
    return sum_rate_utility
def VAE_encoding_loss(k, l):
    def loss_fn(y_pred):
        y_pred_code = y_pred[:, k:]
        y_pred_z_q = y_pred_code[:, :k*l]
        y_pred_z_e = y_pred_code[:, k*l:]
        # quantization_loss_1 = tf.square(tf.norm(tf.stop_gradient(y_pred_z_e)- y_pred_z_q))/y_pred.shape[0]]
        quantization_loss_1 = tf.losses.mean_squared_error(y_pred_z_q, tf.stop_gradient(y_pred_z_e))
        # quantization_loss_2 = 0.25*tf.square(tf.norm(tf.stop_gradient(y_pred_z_q) - y_pred_z_e))/y_pred.shape[0]
        quantization_loss_2 = 0.25*tf.losses.mean_squared_error(y_pred_z_e, tf.stop_gradient(y_pred_z_q))
        return quantization_loss_1 + quantization_loss_2
    return loss_fn
class VAE_loss_general():
    def __init__(self, moving_avg = False, model = None):
        self.moving_avg = moving_avg
        self.model = model
        self.M = None
        self.N = None
    def call(self, z_q, z_e):
        # quantization_loss_1 = tf.square(tf.norm(tf.stop_gradient(y_pred_z_e)- y_pred_z_q))/y_pred.shape[0]]
        quantization_loss_1 = tf.losses.mean_squared_error(z_q, tf.stop_gradient(z_e))
        # quantization_loss_2 = 0.25*tf.square(tf.norm(tf.stop_gradient(y_pred_z_q) - y_pred_z_e))/y_pred.shape[0]
        quantization_loss_2 = tf.losses.mean_squared_error(z_e, tf.stop_gradient(z_q))
        loss = quantization_loss_1 + 0.25 * quantization_loss_2
        if self.moving_avg:
            code_book = self.model.get_layer("Closest_embedding_layer_moving_avg").E
            last_m = self.model.get_layer("Closest_embedding_layer_moving_avg").last_m
            choices = [[] for i in range(last_m.shape[1])]
            distances = (KB.sum(z_e ** 2, axis=2, keepdims=True)
                         - 2 * KB.dot(z_e, code_book)
                         + KB.sum(code_book ** 2, axis=0, keepdims=True))
            encoding_indices = KB.argmax(-distances, axis=2)
            encoding_indices = tf.reshape(encoding_indices,
                                          shape=(encoding_indices.shape[0] * encoding_indices.shape[1],))
            z_e = tf.reshape(z_e, shape=(z_e.shape[0] * z_e.shape[1], z_e.shape[2]))
            for i in range(0, z_e.shape[0]):
                choices[encoding_indices[i].numpy()].append(z_e[i:i+1,:])
                # print(len(choices[encoding_indices[i].numpy()]))
            if self.M is None:
                self.M = [0 for i in range(last_m.shape[1])]
                self.N = [0 for i in range(last_m.shape[1])]
                for i in range(0, len(choices)):
                    if len(choices[i]) != 0:
                        self.M[i] = tf.reduce_sum(tf.concat(choices[i], axis=0), axis=0)
                        self.N[i] = tf.Variable(len(choices[i]), dtype=tf.float32)
                    else:
                        # self.M[i] = None # alternatively adjust M to be the mean of all encoding + some noise
                        self.M[i] = tf.reduce_mean(z_e, axis=0) + tf.random.normal((z_e.shape[1],), 0, tf.math.reduce_std(z_e, axis=0))
                        self.N[i] = tf.Variable(1, dtype=tf.float32)
            else:
                for i in range(0, len(choices)):
                    if len(choices[i]) != 0:
                        self.M[i] = 0.99 * self.M[i] + 0.01 * tf.reduce_sum(tf.concat(choices[i], axis=0), axis=0)
                        self.N[i] = 0.99 * self.N[i] + 0.01 * tf.Variable(len(choices[i]), dtype=tf.float32)
                    else:
                        self.M[i] = 0.99 * self.M[i] + 0.01 * tf.random.normal((z_e.shape[1],), 0, tf.math.reduce_std(z_e, axis=0))
                        self.N[i] = 0.99 * self.N[i] + 0.01 * tf.Variable(1, dtype=tf.float32)
                        # self.M[i] = None
                    # if len(choices[i]) != 0 and not (self.M[i] is None):
                    #     self.M[i] = 0.99*self.M[i] + 0.01*tf.reduce_sum(tf.concat(choices[i], axis=0), axis=0)
                    #     self.N[i] = 0.99*self.N[i] + 0.01*tf.Variable(len(choices[i]), dtype=tf.float32)
                    # elif len(choices[i]) != 0 and self.M[i] is None:
                    #     self.M[i] = tf.reduce_sum(tf.concat(choices[i], axis=0), axis=0)
                    #     self.N[i] = tf.Variable(len(choices[i]), dtype=tf.float32)
                    # else:
                    #     # self.M[i] = None
                    #     self.M[i] = tf.reduce_mean(z_e, axis=0) + tf.random.normal((z_e.shape[1],), 0, 0.01)
            for i in range(0, code_book.shape[1]):
                if self.N[i] > 0:
                    self.model.get_layer("Closest_embedding_layer_moving_avg").E[:, i].assign(self.M[i]/self.N[i])
        return tf.reduce_mean(loss, axis=1)
def Reconstruction_loss():
    def loss_fn(reconstructed_input, features):
        features = tf.abs(features)
        distance = tf.math.reduce_euclidean_norm(reconstructed_input-features, axis=2)
        loss = tf.reduce_mean(distance, axis=1)
        return loss
    return loss_fn
def All_softmaxes_CE(N_rf):
    def loss_fn(per_user_softmaxes, overall_softmax):
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(tf.argmax(overall_softmax, axis=2),
                                                                        overall_softmax)
        mask = tf.stop_gradient(Harden_scheduling(k=N_rf)(per_user_softmaxes))
        loss = loss + tf.keras.losses.CategoricalCrossentropy()(per_user_softmaxes, mask)
        return loss
    return loss_fn
def All_softmaxes_CE_general(N_rf, K, M):
    def loss_fn(raw_output):
        loss = 0
        for i in range(0, N_rf):
            loss = loss + tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(tf.argmax(raw_output[:, :, i], axis=1), raw_output[:, :, i])
            # loss = loss + tf.keras.losses.MeanSquaredError()(tf.argmax(raw_output[:, :, i], axis=1), raw_output[:, :, i])
        return loss
    return loss_fn
def All_softmaxes_MSE_general(N_rf, K, M):
    def loss_fn(raw_output):
        loss = 0
        for i in range(0, N_rf):
            # loss = loss + tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(tf.argmax(raw_output[:, :, i], axis=1), raw_output[:, :, i])
            # loss = loss + tf.keras.losses.MeanSquaredError()(tf.argmax(raw_output[:, :, i], axis=1), raw_output[:, :, i])
            loss = loss + tf.square(1.0 - tf.reduce_max(raw_output[:, :, i], axis=1))
            loss = loss + tf.reduce_sum(tf.reduce_sum(tf.square(0.0 - raw_output[:, :, i]), axis=1) - tf.square(0.0 - tf.reduce_max(raw_output[:, :, i], axis=1)))
        return loss/(1.0*N_rf)/(1.0*K)/(1.0*M)
    return loss_fn


# =========================== Custom function for straight through estimation ============================
@tf.custom_gradient
def sign_relu_STE(x):
    rtv = tf.sign(x)
    def grad(dy):
        one = tf.constant(1, dtype=tf.float32)
        zero = tf.constant(0, dtype=tf.float32)
        # back = tf.maximum(tf.minimum(x, pos), neg)
        back = tf.where(tf.abs(x) >= 0, one, zero)
        grad_val = dy * back
        return grad_val
    return rtv, grad
def binary_activation(x, shift=0.5):
    out = tf.maximum(tf.sign(x-shift), 0)
    return out
def hard_tanh(x):
    neg = tf.constant(-1, dtype=tf.float32)
    pos = tf.constant(1, dtype=tf.float32)
    rtv = tf.maximum(tf.minimum(x, pos), neg)
    return rtv
def leaky_hard_sigmoid(x):
    rtv = tf.maximum(0.0, tf.minimum(x, 0.01*(x-1.0) + 1.0))
    return rtv
def hard_sigmoid(x):
    zero = tf.constant(0, dtype=tf.float32)
    pos = tf.constant(1, dtype=tf.float32)
    rtv = tf.maximum(tf.minimum(x, pos), zero)
    return rtv
def multi_level_thresholding(x, l):
    level = tf.constant(l-1, dtype=tf.float32)
    rtv = tf.round(x*level)/level
    return rtv
def leaky_hard_tanh(x):
    rtv = tf.maximum(tf.minimum(x, 1.0 + 0.01 * x), -1.0 + 0.01 * x)
    return rtv
def clippedRelu(x):
    return tf.maximum(tf.minimum(0.01 * x, x), 0.01 * x)
def annealing_sigmoid(x, N):
    alpha = tf.minimum(5.0, 1.0 + 0.01*N)
    out = tf.sigmoid(alpha*x)
    return out
def annealing_tanh(x, N, name):
    alpha = tf.minimum(5.0, 1.0 + 0.01*N)
    out = tf.tanh(alpha*x, name=name)
    return out
# ========================================== Layers ==========================================
class SubtractLayer(tf.keras.layers.Layer):
    def __init__(self, name):
      super(SubtractLayer, self).__init__(name=name)
      self.thre = tf.Variable(tf.constant([0.5,0.5], shape=[2,1]), trainable=True, name="threshold")
    def call(self, inputs):
      return inputs - self.thre
class SubtractLayer_with_noise(tf.keras.layers.Layer):
    def __init__(self, name):
      super(SubtractLayer_with_noise, self).__init__(name=name)
      self.noise = tf.constant(tf.random.uniform(shape=[2,1], minval=-0.01, maxval=0.011))
      self.thre = tf.Variable(tf.constant([0.5,0.5], shape=[2,1]), trainable=True, name="threshold")
    def call(self, inputs):
      return inputs - self.thre + tf.random.normal(shape=[2, 1], mean=0, stddev=self.noise)

@tf.custom_gradient
def STE_argmax(x):
    # assuming
    top_val = tf.tile(tf.reduce_max(x, axis=1, keepdims=True), [1, x.shape[1], 1])
    result = tf.where(x == top_val, 1.0, 0.0)
    def grad(dy):
        return dy
    return result, grad
class Argmax_STE_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(Argmax_STE_layer, self).__init__()
    def call(self, x):
        return STE_argmax(x)
@tf.custom_gradient
def SPIGOT_argmax(x):
    # assuming
    top_val = tf.tile(tf.reduce_max(x, axis=1, keepdims=True), [1, x.shape[1], 1])
    result = tf.where(x == top_val, 1.0, 0.0)
    def grad(dy):
        p_hat = x - 0.01 * dy
        top_val_p_hat = tf.tile(tf.reduce_max(p_hat, axis=1, keepdims=True), [1, x.shape[1], 1])
        z_tilde = tf.where(p_hat == top_val_p_hat, 1.0, 0.0)
        return z_tilde - result
    return result, grad
class Argmax_SPIGOT_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(Argmax_SPIGOT_layer, self).__init__()
    def call(self, x):
        return SPIGOT_argmax(x)
class Sparsemax(tf.keras.layers.Layer):
    """Sparsemax activation function.
    The output shape is the same as the input shape.
    See [From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification](https://arxiv.org/abs/1602.02068).
    Arguments:
        axis: Integer, axis along which the sparsemax normalization is applied.
    """

    def __init__(self, axis: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs):
        return sparsemax(inputs, axis=self.axis)

    def get_config(self):
        config = {"axis": self.axis}
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape
# @tf.keras.utils.register_keras_serializable(package="Addons")
def sparsemax(logits, axis: int = -1) -> tf.Tensor:
    """Sparsemax activation function [1].
    For each batch `i` and class `j` we have
      $$sparsemax[i, j] = max(logits[i, j] - tau(logits[i, :]), 0)$$
    [1]: https://arxiv.org/abs/1602.02068
    Args:
        logits: Input tensor.
        axis: Integer, axis along which the sparsemax operation is applied.
    Returns:
        Tensor, output of sparsemax transformation. Has the same type and
        shape as `logits`.
    Raises:
        ValueError: In case `dim(logits) == 1`.
    """
    logits = tf.convert_to_tensor(logits, name="logits")

    # We need its original shape for shape inference.
    shape = logits.get_shape()
    rank = shape.rank
    is_last_axis = (axis == -1) or (axis == rank - 1)

    if is_last_axis:
        output = _compute_2d_sparsemax(logits)
        output.set_shape(shape)
        return output

    # If dim is not the last dimension, we have to do a transpose so that we can
    # still perform softmax on its last dimension.

    # Swap logits' dimension of dim and its last dimension.
    rank_op = tf.rank(logits)
    axis_norm = axis % rank
    logits = _swap_axis(logits, axis_norm, tf.math.subtract(rank_op, 1))

    # Do the actual softmax on its last dimension.
    output = _compute_2d_sparsemax(logits)
    output = _swap_axis(output, axis_norm, tf.math.subtract(rank_op, 1))

    # Make shape inference work since transpose may erase its static shape.
    output.set_shape(shape)
    return output
def _swap_axis(logits, dim_index, last_index, **kwargs):
    return tf.transpose(
        logits,
        tf.concat(
            [
                tf.range(dim_index),
                [last_index],
                tf.range(dim_index + 1, last_index),
                [dim_index],
            ],
            0,
        ),
        **kwargs,
    )
def _compute_2d_sparsemax(logits):
    """Performs the sparsemax operation when axis=-1."""
    shape_op = tf.shape(logits)
    obs = tf.math.reduce_prod(shape_op[:-1])
    dims = shape_op[-1]

    # In the paper, they call the logits z.
    # The mean(logits) can be substracted from logits to make the algorithm
    # more numerically stable. the instability in this algorithm comes mostly
    # from the z_cumsum. Substacting the mean will cause z_cumsum to be close
    # to zero. However, in practise the numerical instability issues are very
    # minor and substacting the mean causes extra issues with inf and nan
    # input.
    # Reshape to [obs, dims] as it is almost free and means the remanining
    # code doesn't need to worry about the rank.
    z = tf.reshape(logits, [obs, dims])

    # sort z
    z_sorted, _ = tf.nn.top_k(z, k=dims)

    # calculate k(z)
    z_cumsum = tf.math.cumsum(z_sorted, axis=-1)
    k = tf.range(1, tf.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
    z_check = 1 + k * z_sorted > z_cumsum
    # because the z_check vector is always [1,1,...1,0,0,...0] finding the
    # (index + 1) of the last `1` is the same as just summing the number of 1.
    k_z = tf.math.reduce_sum(tf.cast(z_check, tf.int32), axis=-1)

    # calculate tau(z)
    # If there are inf values or all values are -inf, the k_z will be zero,
    # this is mathematically invalid and will also cause the gather_nd to fail.
    # Prevent this issue for now by setting k_z = 1 if k_z = 0, this is then
    # fixed later (see p_safe) by returning p = nan. This results in the same
    # behavior as softmax.
    k_z_safe = tf.math.maximum(k_z, 1)
    indices = tf.stack([tf.range(0, obs), tf.reshape(k_z_safe, [-1]) - 1], axis=1)
    tau_sum = tf.gather_nd(z_cumsum, indices)
    tau_z = (tau_sum - 1) / tf.cast(k_z, logits.dtype)

    # calculate p
    p = tf.math.maximum(tf.cast(0, logits.dtype), z - tf.expand_dims(tau_z, -1))
    # If k_z = 0 or if z = nan, then the input is invalid
    p_safe = tf.where(
        tf.expand_dims(
            tf.math.logical_or(tf.math.equal(k_z, 0), tf.math.is_nan(z_cumsum[:, -1])),
            axis=-1,
        ),
        tf.fill([obs, dims], tf.cast(float("nan"), logits.dtype)),
        p,
    )

    # Reshape back to original size
    p_safe = tf.reshape(p_safe, shape_op)
    return p_safe

# ========================================== MISC ==========================================
def random_complex(shape, sigma2):
    A_R = tf.random.normal(shape, 0, sigma2, dtype=tf.float32)
    A_I = tf.random.normal(shape, 0, sigma2, dtype=tf.float32)
    A = tf.complex(A_R, A_I)

    return A
def quantization_evaluation(model, granuality = 0.001, k=2, saveImg = False, name="", bitstring=True):
    dim_num = int(1/granuality) + 1
    count = np.arange(0, 1 + granuality, granuality)
    output = np.zeros((dim_num, dim_num))
    input = np.zeros((dim_num*dim_num, 2))
    line = 0
    for i in range(count.shape[0]):
        for j in range(count.shape[0]):
            input[line, 0] = count[i]
            input[line, 1] = count[j]
            line = line + 1
    channel_data = tf.constant(input)
    channel_label = tf.math.argmax(channel_data, axis=1)
    if bitstring:
        channel_data = float_to_floatbits(channel_data)
    ds = Dataset.from_tensor_slices((channel_data, channel_label)).batch(dim_num*dim_num)
    for features, labels in ds:
        if bitstring:
            features_mod = tf.ones((features.shape[0], features.shape[1], 1)) * 1
            features_mod = tf.concat((features_mod, features), axis=2)
        else:
            # features_mod = tf.ones((features.shape[0], features.shape[1], 1), dtype=tf.float64)
            # features_mod = tf.concat((features_mod, tf.reshape(features, (features.shape[0], features.shape[1], 1))), axis=2)
            pass
        out = model(features)
        out = out[:, :2]
        prediction = tf.argmax(out, axis=1)
    line = 0
    for i in range(count.shape[0]):
        for j in range(count.shape[0]):
            if prediction[line] == labels[line]:
                output[i, j] = 1
            line = line + 1
    if saveImg == False:
        plot_quantization_square(output, granuality)
    else:
        save_quantization_square(output, granuality, name)
def save_quantization_square(output, granuality, name):
    dim_num = int(1 / granuality) + 1
    count = np.arange(0, 1 + granuality, granuality)
    output = np.flip(output, axis=0)
    plt.imshow(output, cmap="gray")
    step_x = int(dim_num / (5 - 1))
    x_positions = np.arange(0, dim_num, step_x)
    x_labels = count[::step_x]
    y_labels = np.array([1, 0.75, 0.5, 0.25, 0])
    plt.xticks(x_positions, x_labels)
    plt.yticks(x_positions, y_labels)
    plt.savefig(name)
def plot_quantization_square(output, granuality):
    dim_num = int(1 / granuality) + 1
    count = np.arange(0, 1 + granuality, granuality)
    output = np.flip(output, axis=0)
    plt.imshow(output, cmap="gray")
    step_x = int(dim_num / (5 - 1))
    x_positions = np.arange(0, dim_num, step_x)
    x_labels = count[::step_x]
    y_labels = np.array([1, 0.75, 0.5, 0.25, 0])
    plt.xticks(x_positions, x_labels)
    plt.yticks(x_positions, y_labels)
    plt.show()
def replace_tanh_with_sign(model, model_func, k):
    model.save_weights('weights.hdf5')
    new_model = model_func((k, ), k, saved=True)
    new_model.load_weights('weights.hdf5')
    return new_model
def make_video(path, training_data):
    images = []
    i = -1
    file_name_temp = "trained_models/Jul 6th/k=2 no bitstring/2_user_1_qbit_threshold_encoder_tanh(relu)_seed=0"
    for i in range(0, 1000):
        # if filename[-3:] == "png":
        i = i + 1
        filename = file_name_temp + str(i) + ".png"
        print(filename)
        acc = np.round(training_data[i, 1]*100)/100
        img = cv2.imread(filename)
        cv2.putText(img, "epochs: " + str(i), org=(20, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255))
        cv2.putText(img, "accuracy: " + str(acc), org=(20, 120), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0))
        height, width, layers = img.shape
        size = (width, height)
        images.append(img)
    out = cv2.VideoWriter('DNN_no_biststring.mpg', cv2.VideoWriter_fourcc(*'XVID'), 15, size)

    for i in range(len(images)):
        out.write(images[i])
    out.release()
def floatbits_to_float(value_arr):
    np_value_arr = value_arr.numpy()
    if len(value_arr.shape) == 3:
        out = np.zeros((value_arr.shape[0], value_arr.shape[1]))
        for i in range(0, 23):
            out = out + np_value_arr[:, :, i] * np.float_power(2, -(i+1))
        return tf.constant(out, dtype=tf.float32)
def float_to_floatbits(value_arr, complex=False):
    cp_value_arr = value_arr.numpy()
    # I'm not sure if the shape thing works well so might have to comeback and fix it
    if complex:
        real = tf.math.real(value_arr)
        imag = tf.math.imag(value_arr)
        real_f = float_to_floatbits(real)
        imag_f = float_to_floatbits(imag)
        return tf.concat([imag_f, real_f], axis=2)
    elif len(value_arr.shape) == 1:
        out = np.zeros((value_arr.shape[0], 23))
        for i in range(0, 23):
            cp_value_arr = cp_value_arr*2
            out[:, i] = np.where(cp_value_arr >= 1, 1, 0)
            cp_value_arr = cp_value_arr - out[:, i]
        return tf.constant(out, dtype=tf.float32)
    elif len(value_arr.shape) == 2:
        out = np.zeros((value_arr.shape[0], value_arr.shape[1], 23))
        for j in range(0, value_arr.shape[1]):
            for i in range(0, 23):
                cp_value_arr[:, j] = cp_value_arr[:, j] * 2
                out[:, j, i] = np.where(cp_value_arr[:, j] >= 1, 1, 0)
                cp_value_arr[:, j] = cp_value_arr[:, j]- out[:, j, i]
        return tf.constant(out, dtype=tf.float32)
    elif len(value_arr.shape) == 3:
        out = np.zeros((value_arr.shape[0], value_arr.shape[1], value_arr.shape[2], 23), dtype=np.float32)
        for i in range(0, 23):
            cp_value_arr[:, :, :] = cp_value_arr[:, :, :] * 2
            out[:, :, :, i] = np.where(cp_value_arr[:, :, :] >= 1, 1.0, 0.0)
            cp_value_arr[:, :, :] = cp_value_arr[:, :, :]- out[:, :, :, i]
        out = tf.reshape(out, [value_arr.shape[0], value_arr.shape[1], value_arr.shape[2]*23])
        return tf.constant(out, dtype=tf.float32)
def freeze_decoder_layers(model):
    for layer in model.layers:
        if layer.name[0:7] == "decoder":
            layer.trainable = False
        elif layer.name[0:7] == "encoder":
            layer.trainable = True
    return model
def generate_binary_encoding(dim):
    encoding_space = np.zeros((2**dim, dim))
    num_range = np.arange(0, 2**dim)
    for i in range(0, encoding_space.shape[1]):
        encoding_space[:, dim-i-1] = num_range%2
        num_range = np.floor(num_range/2)
    return tf.constant(encoding_space, dtype=tf.float32)
def sparse_matrix_from_full(G, p):
    G = (tf.abs(G))
    K = G.shape[1]
    M = G.shape[2]
    top_values, top_indices = tf.math.top_k(G, k=p)
    G_copy = np.zeros((top_indices.shape[0], K, M))
    for n in range(0, top_indices.shape[0]):
        for i in range(0, K * p):
            # print(K*p)
            p_i = int(i % p)
            user_i = int(tf.floor(i / p))
            G_copy[n, user_i, int(top_indices[n, user_i, p_i])] = top_values[n, user_i, p_i]
    G_copy = tf.constant(G_copy, dtype=tf.float32)
    # print(top_indices.shape)
    # mask = tf.scatter_nd(top_indices, tf.expand_dims(G, 3), tf.shape(G))
    # print(mask.shape)
    # A[2]
    return G_copy
if __name__ == "__main__":
    N = 500
    M = 64
    K = 50
    B = 5
    Nrf = 7
    sigma2 = 0
    data = generate_link_channel_data(1, K, M, Nrf=1)
    A[2]

    data = tf.square(tf.abs(data[0]))
    gsm = GumbelSoftmax(0.3, logits=True)
    tim = tf.random.normal((5, ))
    print(tim)
    print(gsm._log_prob(tim))
    A[2]
    print(STE_argmax(tim, -1, [1,1,5]))
    A[1]
    data = data/tf.reduce_max(data)
    print(data.shape)
    for i in range(0, K):
        print(tf.reduce_mean(data[i]), tf.reduce_max(data[i]), tf.reduce_min(data[i]))
    A[2]
    # model = FDD_encoding_model_constraint_123_with_softmax_and_ranking(M, K, B)
    # features = generate_link_channel_data(N, K, M)
    # predictions = model(features)
    # loss_1 = Sum_rate_utility_top_k_with_mask(K, M, sigma2)(predictions, features)

