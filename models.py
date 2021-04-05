import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LeakyReLU, Softmax, Input, ThresholdedReLU, Flatten
from tensorflow.keras.activations import sigmoid
import random
from tensorflow.keras import backend as KB
from util import *
# import tensorflow_addons as tfa
# from sklearn.cluster import KMeans
# from matplotlib import pyplot as plt
############################## Trained Loss Functions ##############################
def MLP_loss_function(inputshape=[1000, 3]):
    inputs = Input(shape=inputshape)
    x = tf.keras.layers.Reshape((3000,))(inputs)
    x = Dense(500)(x)
    x = LeakyReLU()(x)
    x = Dense(500)(x)
    x = LeakyReLU()(x)
    x = Dense(1)(x)
    model = Model(inputs, x, name="category_count_MLP")
    return model
def LSTM_loss_function(k, input_shape=[], state_size=30):
    inputs = Input(shape=input_shape)
    x = tf.keras.layers.LSTM(state_size)(inputs)
    x = Dense(60)(x)
    x = LeakyReLU()(x)
    x = Dense(20)(x)
    x = LeakyReLU()(x)
    x = Dense(1)(x)
    model = Model(inputs, x, name="category_count_LSTM")
    return model
def Convnet_loss_function(input_shape=[1000, 4], combinations=8):
    # you need to format the input into (1, batchsize, encoding size)
    inputs = Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(filters=combinations * 2, kernel_size=input_shape[1], strides=1,
                               name="variation_scanners_1")(inputs)
    x = tf.keras.layers.MaxPool1D(pool_size=10)(x)
    x = tf.keras.layers.Conv1D(filters=combinations * 2, kernel_size=input_shape[1], strides=1,
                               name="variation_scanners_2")(x)
    x = tf.keras.layers.MaxPool1D(pool_size=5)(x)
    x = tf.keras.layers.Reshape([combinations * 19 * 2])(x)
    x = tf.keras.layers.Dense(200)(x)
    x = sigmoid(x)
    x = tf.keras.layers.Dense(200)(x)
    x = sigmoid(x)
    x = tf.keras.layers.Dense(1)(x)
    x = LeakyReLU()(x)
    model = Model(inputs, x, name="category_count_conv_net")
    print(model.summary())
    return model
############################## analytical model ##############################
def create_uniformed_quantization_model(k, bin_num=10, prob=True):
    def uniformed_quantization_prob(x):
        x = tf.round(x * bin_num) / bin_num
        max_x = tf.argmax(x, axis=1).numpy()
        max_x = max_x.flatten()
        col = np.arange(0, x.shape[0])
        out = np.zeros(x.shape)
        out[col, max_x] = 1
        out = tf.convert_to_tensor(out)
        return out

    def uniformed_quantization_reg(x):
        x = tf.round(x * bin_num) / bin_num
        max_x = tf.argmax(x, axis=1).numpy()
        return max_x

    if prob:
        return uniformed_quantization_prob
    else:
        return uniformed_quantization_reg
def create_optimal_model_k_2(k, input):
    x = floatbits_to_float(input)
def k_clustering_hieristic(N_rf):
    def model(G, angle=-1):
        G = tf.abs(G).numpy()
        G_original = G.copy()
        for n in range(0, G.shape[0]):
            for k in range(0, G.shape[1]):
                G[n, k] = (G[n, k] - G[n, k].min()) / (G[n, k].max() - G[n, k].min())
        output = np.zeros((G.shape[0], G.shape[1] * G.shape[2]))
        # kmeans_tot = KMeans(n_clusters=N_rf, random_state=0).fit(G[0:500].reshape(500*G.shape[1], G.shape[2]))
        for n in range(0, G.shape[0]):
            # kmeans = kmeans_tot.predict(G[n])
            kmeans = KMeans(n_clusters=N_rf, random_state=0).fit_predict(G[n])
            clusters = []
            for i in range(0, N_rf):
                clusters.append(np.zeros(G[0].shape))
            for i in range(G.shape[1]):
                clusters[kmeans[i]][i] = (G_original[n, i])
            for i in range(0, N_rf):
                max = int(np.argmax(clusters[i]))
                if np.sum(clusters[i]) != 0:
                    output[n, max] = 1
                    G_original[n, int(max / G.shape[2]), max % G.shape[2]] = 0
            for i in range(0, N_rf):
                if np.sum(clusters[i]) == 0:
                    max = int(np.argmax(G_original[n]))
                    output[n, max] = 1
                    G_original[n, int(max / G.shape[2]), max % G.shape[2]] = 0
            # visualize
            visualize = False
            visualize_angle = False
            if visualize:
                img = np.zeros((G.shape[1], G.shape[2], 3))
                colors = {0: np.array([1, 0, 0]), 1: np.array([0, 1, 0]), 2: np.array([0, 0, 1]),
                          3: np.array([1, 0, 1]), 4: np.array([0, 1, 1])}
                for i in range(0, G.shape[1]):
                    img[i, :] = colors[kmeans[i]]
                G[n] = (G[n] - G[n].min()) / (G[n].max() - G[n].min())
                for i in range(0, 3):
                    img[:, :, i] = img[:, :, i] * G[n]
                plt.imshow(img)
                plt.show()
            if visualize_angle:
                colors = {0: np.array([1, 0, 0]), 1: np.array([0, 1, 0]), 2: np.array([0, 0, 1]),
                          3: np.array([1, 0, 1]), 4: np.array([0, 1, 1])}
                colorss = []
                for i in range(0, len(angle[n])):
                    print(angle)
                    colorss.append(colors[kmeans[i]])
                plt.scatter(np.sin(angle[n]), np.ones(angle[n].shape), c=colorss)
                plt.show()
        return output

    return model
def top_N_rf_user_model(M, K, N_rf):
    def model(G):
        G = tf.reshape(G, (G.shape[0], M * K))
        G_flat = tf.square(tf.abs(G)).numpy()
        out = np.zeros(G_flat.shape)
        for i in range(0, K):
            max = np.argmax(G_flat[:, M * i:M * (i + 1)], axis=1)
            out[:, M * i + max] = 1
        G_with_precoder = out * G_flat
        out_2 = np.zeros(out.shape)
        values, index = tf.math.top_k(G_with_precoder, k=N_rf)
        for i in range(0, G_with_precoder.shape[0]):
            out_2[i, index[i]] = 1
        return tf.constant(out_2, dtype=tf.float32)

    return model
def PF_DFT_model(M, K, N_rf , sigma2):
    def model(G, weight):
        # G = tf.reshape(G, (G.shape[0], M * K))
        G_flat = tf.square(tf.abs(G)).numpy()
        out = np.zeros((G.shape[0], M * K))

        G_best_precoder = np.argmax(G_flat, axis=2)
        for n in range(0, G.shape[0]):
            choices = {}
            for k in range(0, K):
                try:
                    current_best_k, current_best, m = choices[G_best_precoder[n, k]]
                    val = np.log2(1 + G_flat[n, k, G_best_precoder[n, k]]/sigma2) * weight[n, k]
                    if (val > current_best):
                        choices[G_best_precoder[n, k]] = [k, val, G_best_precoder[n, k]]
                except:
                    choices[G_best_precoder[n, k]] = [k, np.log2(1 + G_flat[n, k, G_best_precoder[n, k]]/sigma2) * weight[n, k], G_best_precoder[n, k]]
            vals = list(choices.values())
            vals = sorted(vals, key=lambda tup: tup[1], reverse=True)
            for i in range(0, min(N_rf, len(vals))):
                out[n, vals[i][0] * M + vals[i][2]] = 1
        # for i in range(0, K): # for each user, set
        #     max = np.argmax(G_flat[:, M * i:M * (i + 1)], axis=1)
        #     out[:, M * i + max] = 1
        # G_with_precoder = out * G_flat
        # out_2 = np.zeros(out.shape)
        # values, index = tf.math.top_k(G_with_precoder, k=N_rf)
        # for i in range(0, G_with_precoder.shape[0]):
        #     out_2[i, index[i]] = 1
        return tf.constant(out, dtype=tf.float32)
    return model
def greedy_hieristic(N_rf, sigma2):
    def model(G):
        combinations = []
        loss = Sum_rate_utility_WeiCui(G.shape[1], G.shape[2], sigma2)
        for i_1 in range(0, G.shape[1] * G.shape[2]):
            for i_2 in range(0, G.shape[1] * G.shape[2]):
                if i_1 != i_2 and tf.floor(i_1 / G.shape[2]) != tf.floor(i_2 / G.shape[2]):
                    temp = np.zeros((G.shape[1] * G.shape[2],))
                    temp[i_1] = 1
                    temp[i_2] = 1
                    combinations.append(temp)
        val_G = tf.abs(G)
        output = np.zeros((val_G.shape[0], val_G.shape[1] * val_G.shape[2]))
        for n in range(G.shape[0]):
            print("number", n)
            min = 100
            best_pair = None
            for com in combinations:
                current_min = loss(tf.expand_dims(tf.constant(com, tf.float32), 0), val_G[n:n + 1])
                if current_min < min:
                    min = current_min
                    best_pair = com
            output[n] = best_pair
            selected = set()
            pair_index = np.nonzero(best_pair)
            selected.add(int(tf.floor(pair_index[0][0] / G.shape[2])))
            selected.add(int(tf.floor(pair_index[0][1] / G.shape[2])))
            if N_rf > 2:
                for n_rf in range(2, N_rf):
                    new_comb = []
                    for additional_i in range(G.shape[1] * G.shape[2]):
                        if output[n, additional_i] != 1 and not int(tf.floor(additional_i / G.shape[2])) in selected:
                            temp = output[n].copy()
                            temp[additional_i] = 1
                            new_comb.append(temp)
                    min = 100
                    best_comb = None
                    for com in new_comb:
                        current_min = loss(tf.expand_dims(tf.constant(com, tf.float32), 0), val_G[n:n + 1])
                        if current_min < min:
                            min = current_min
                            best_comb = com
                    pair_index = np.nonzero(best_comb)[0]
                    for each_nrf in range(0, pair_index.shape[0]):
                        selected.add(int(tf.floor(pair_index[each_nrf] / G.shape[2])))
                    output[n] = best_comb
        output = tf.constant(output, dtype=tf.float32)
        return output

    return model
def sparse_pure_greedy_hueristic(N_rf, sigma2, K, M, p):
    def model(top_val, top_indice, G=None):
        loss = Sum_rate_utility_WeiCui(K, M, sigma2)
        output = np.zeros((top_indice.shape[0], K * M))
        if G is None:
            G_copy = np.zeros((top_indice.shape[0], K, M))
            for n in range(0, top_indice.shape[0]):
                for i in range(0, K * p):
                    # print(K*p)
                    p_i = int(i % p)
                    user_i = int(tf.floor(i / p))
                    G_copy[n, user_i, int(top_indice[n, user_i, p_i])] = top_val[n, user_i, p_i]
            G_copy = tf.constant(G_copy, dtype=tf.float32)
            G = G_copy
        print("done generating partial information")
        for n in range(0, G.shape[0]):
            # print("==================================== type", n, "====================================")
            combinations = []
            for index_1 in range(0, K * p):
                p_1 = int(index_1 % p)
                user_1 = int(tf.floor(index_1 / p))
                comb = np.zeros((K * M,))
                comb[user_1 * M + top_indice[n, user_1, p_1]] = 1
                combinations.append(comb)
            min = 100
            best_one = None
            for com in combinations:
                current_min = loss(tf.expand_dims(tf.constant(com, tf.float32), 0), G[n:n + 1])
                if current_min < min:
                    min = current_min
                    best_one = com
            output[n] = best_one
            selected = set()
            pair_index = np.nonzero(best_one)
            selected.add(int(tf.floor(pair_index[0][0] / G.shape[2])))
            if N_rf >= 2:
                for n_rf in range(1, N_rf):
                    new_comb = []
                    for additional_i in range(0, K * p):
                        p_i = int(additional_i % p)
                        user_i = int(tf.floor(additional_i / p))
                        beamformer_i = int(top_indice[n, user_i, p_i])
                        if output[n, user_i * M + beamformer_i] != 1:
                            if not int(tf.floor((user_i * M + beamformer_i) / M)) in selected:
                                temp = output[n].copy()
                                temp[user_i * M + beamformer_i] = 1
                                new_comb.append(temp)
                    min = 100
                    best_comb = None
                    for com in new_comb:
                        current_min = loss(tf.expand_dims(tf.constant(com, tf.float32), 0), G[n:n + 1])
                        if current_min < min:
                            min = current_min
                            best_comb = com
                    output[n] = best_comb
                    pair_index = np.nonzero(best_comb)[0]
                    for each_nrf in range(0, pair_index.shape[0]):
                        selected.add(int(tf.floor(pair_index[each_nrf] / G.shape[2])))
        output = tf.constant(output, dtype=tf.float32)
        return output

    return model
def best_weight_model(N_rf, K, M):
    def model(env:Weighted_sumrate_model, G,N_rf=N_rf, K=K, M=M):
        weights = env.get_weight()
        top_values_k, top_indices_k = tf.math.top_k(weights, k=N_rf)
        top_values_m, top_indices_m = tf.math.top_k(np.abs(G), k=1)
        out = np.zeros((G.shape)).reshape(G.shape[0], K*M)
        for n in range(0, G.shape[0]):
            for i in range(0, N_rf):
                out[n, top_indices_k[n, i] * M + top_indices_m[n, top_indices_k[n, i]]] = 1
        out = np.array(out, dtype = np.float32)
        return tf.convert_to_tensor(out, dtype=tf.float32)
    return model
# this one do an exhastive for Nrf = 2, then do Greedy upwards
def sparse_greedy_hueristic(N_rf, sigma2, K, M, p):
    def model(top_val, top_indice):
        loss = Sum_rate_utility_WeiCui(K, M, sigma2)
        output = np.zeros((top_indice.shape[0], K * M))
        G_copy = np.zeros((top_indice.shape[0], K, M))
        for n in range(0, top_indice.shape[0]):
            for i in range(0, K * p):
                # print(K*p)
                p_i = int(i % p)
                user_i = int(tf.floor(i / p))
                G_copy[n, user_i, int(top_indice[n, user_i, p_i])] = top_val[n, user_i, p_i]
        G_copy = tf.constant(G_copy, dtype=tf.float32)
        print("done generating partial information")
        for n in range(0, G_copy.shape[0]):
            # print("==================================== type", n, "====================================")
            combinations = []
            count = 0
            for index_1 in range(0, K * p):
                for index_2 in range(0, K * p):
                    p_1 = int(index_1 % p)
                    user_1 = int(tf.floor(index_1 / p))
                    p_2 = int(index_2 % p)
                    user_2 = int(tf.floor(index_2 / p))
                    if index_1 != index_2 and user_1 != user_2:
                        comb = np.zeros((K * M,))
                        comb[user_1 * M + top_indice[n, user_1, p_1]] = 1
                        comb[user_2 * M + top_indice[n, user_2, p_2]] = 1
                        combinations.append(comb)
                    count = count + 1
                if count >= 1000:
                    break
            min = 100
            best_pair = None
            for com in combinations:
                current_min = loss(tf.expand_dims(tf.constant(com, tf.float32), 0), G_copy[n:n + 1])
                if current_min < min:
                    min = current_min
                    best_pair = com
            output[n] = best_pair
            selected = set()
            pair_index = np.nonzero(best_pair)
            selected.add(int(tf.floor(pair_index[0][0] / G_copy.shape[2])))
            selected.add(int(tf.floor(pair_index[0][1] / G_copy.shape[2])))
            if N_rf > 2:
                for n_rf in range(2, N_rf):
                    new_comb = []
                    for additional_i in range(0, K * p):
                        p_i = int(additional_i % p)
                        user_i = int(tf.floor(additional_i / p))
                        beamformer_i = int(top_indice[n, user_i, p_i])
                        if output[n, user_i * M + beamformer_i] != 1:
                            if not int(tf.floor((user_i * M + beamformer_i) / M)) in selected:
                                temp = output[n].copy()
                                temp[user_i * M + beamformer_i] = 1
                                new_comb.append(temp)
                    min = 100
                    best_comb = None
                    for com in new_comb:
                        current_min = loss(tf.expand_dims(tf.constant(com, tf.float32), 0), G_copy[n:n + 1])
                        if current_min < min:
                            min = current_min
                            best_comb = com
                    output[n] = best_comb
                    pair_index = np.nonzero(best_comb)[0]
                    for each_nrf in range(0, pair_index.shape[0]):
                        selected.add(int(tf.floor(pair_index[each_nrf] / G_copy.shape[2])))
        output = tf.constant(output, dtype=tf.float32)
        return output

    return model
def DP_sparse_greedy_hueristic(N_rf, sigma2, K, M, p, prev_Nrf=0, prev_out=None):
    def model(top_val, top_indice):
        loss = Sum_rate_utility_WeiCui(K, M, sigma2)
        output = np.zeros((top_indice.shape[0], K * M))
        G_copy = np.zeros((top_indice.shape[0], K, M))
        for n in range(0, top_indice.shape[0]):
            for i in range(0, K * p):
                # print(K*p)
                p_i = int(i % p)
                user_i = int(tf.floor(i / p))
                G_copy[n, user_i, int(top_indice[n, user_i, p_i])] = top_val[n, user_i, p_i]
        G_copy = tf.constant(G_copy, dtype=tf.float32)
        print("done generating partial information")
        for n in range(0, G_copy.shape[0]):
            # print("==================================== type", n, "====================================")
            selected = set()
            if prev_Nrf < 2:
                combinations = []
                for index_1 in range(0, K * p):
                    for index_2 in range(0, K * p):
                        p_1 = int(index_1 % p)
                        user_1 = int(tf.floor(index_1 / p))
                        p_2 = int(index_2 % p)
                        user_2 = int(tf.floor(index_2 / p))
                        if index_1 != index_2 and user_1 != user_2:
                            comb = np.zeros((K * M,))
                            comb[user_1 * M + top_indice[n, user_1, p_1]] = 1
                            comb[user_2 * M + top_indice[n, user_2, p_2]] = 1
                            combinations.append(comb)
                min = 100
                best_pair = None
                for com in combinations:
                    current_min = loss(tf.expand_dims(tf.constant(com, tf.float32), 0), G_copy[n:n + 1])
                    if current_min < min:
                        min = current_min
                        best_pair = com
                output[n] = best_pair
                pair_index = np.nonzero(best_pair)
                selected.add(int(tf.floor(pair_index[0][0] / G_copy.shape[2])))
                selected.add(int(tf.floor(pair_index[0][1] / G_copy.shape[2])))
            elif prev_Nrf >= 2:
                output[n] = prev_out[n]
                pair_index = np.nonzero(prev_out[n])
                selected.add(int(tf.floor(pair_index[0][0] / G_copy.shape[2])))
                selected.add(int(tf.floor(pair_index[0][1] / G_copy.shape[2])))
            if N_rf > 2:
                for n_rf in range(3, N_rf+1):
                    if n_rf > prev_Nrf:
                        print(prev_Nrf)
                        new_comb = []
                        for additional_i in range(0, K * p):
                            p_i = int(additional_i % p)
                            user_i = int(tf.floor(additional_i / p))
                            beamformer_i = int(top_indice[n, user_i, p_i])
                            if output[n, user_i * M + beamformer_i] != 1:
                                if not int(tf.floor((user_i * M + beamformer_i) / M)) in selected:
                                    temp = output[n].copy()
                                    temp[user_i * M + beamformer_i] = 1
                                    new_comb.append(temp)
                        min = 100
                        best_comb = None
                        for com in new_comb:
                            current_min = loss(tf.expand_dims(tf.constant(com, tf.float32), 0), G_copy[n:n + 1])
                            if current_min < min:
                                min = current_min
                                best_comb = com
                        output[n] = best_comb
                        pair_index = np.nonzero(best_comb)[0]
                        for each_nrf in range(0, pair_index.shape[0]):
                            selected.add(int(tf.floor(pair_index[each_nrf] / G_copy.shape[2])))
                    else:
                        output[n] = prev_out[n]
                        pair_index = np.nonzero(prev_out[n])[0]
                        for each_nrf in range(0, pair_index.shape[0]):
                            selected.add(int(tf.floor(pair_index[each_nrf] / G_copy.shape[2])))
        output = tf.constant(output, dtype=tf.float32)
        return output

    return model
def DP_sparse_pure_greedy_hueristic(N_rf, sigma2, K, M, p, G, prev_Nrf=0, prev_out=None):
    def model(top_val, top_indice):
        loss = Sum_rate_utility_WeiCui(K, M, sigma2)
        output = np.zeros((top_indice.shape[0], K * M))
        G_copy = G
        for n in range(0, G_copy.shape[0]):
            # print("==================================== type", n, "====================================")
            selected = set()
            combinations = []
            if prev_Nrf == 0:
                for index_1 in range(0, K * p):
                    p_1 = int(index_1 % p)
                    user_1 = int(tf.floor(index_1 / p))
                    comb = np.zeros((K * M,))
                    comb[user_1 * M + top_indice[n, user_1, p_1]] = 1
                    combinations.append(comb)
                min = 100
                best_one = None
                for com in combinations:
                    current_min = loss(tf.expand_dims(tf.constant(com, tf.float32), 0), G[n:n + 1])
                    if current_min < min:
                        min = current_min
                        best_one = com
                # print(min)
                output[n] = best_one
                selected = set()
                pair_index = np.nonzero(best_one)
                selected.add(int(tf.floor(pair_index[0][0] / G.shape[2])))
            if prev_Nrf >= 1:
                output[n] = prev_out[n]
                pair_index = np.nonzero(prev_out[n])
                selected.add(int(tf.floor(pair_index[0][0] / G_copy.shape[2])))
            if N_rf >= 2:
                for n_rf in range(2, N_rf+1):
                    if n_rf > prev_Nrf:
                        new_comb = []
                        for additional_i in range(0, K * p):
                            p_i = int(additional_i % p)
                            user_i = int(tf.floor(additional_i / p))
                            beamformer_i = int(top_indice[n, user_i, p_i])
                            if output[n, user_i * M + beamformer_i] != 1:
                                if not int(tf.floor((user_i * M + beamformer_i) / M)) in selected:
                                    temp = output[n].copy()
                                    temp[user_i * M + beamformer_i] = 1
                                    new_comb.append(temp)
                        min = 100
                        best_comb = None
                        for com in new_comb:
                            current_min = loss(tf.expand_dims(tf.constant(com, tf.float32), 0), G_copy[n:n + 1])
                            if current_min < min:
                                min = current_min
                                best_comb = com
                        # print(min)
                        output[n] = best_comb
                        pair_index = np.nonzero(best_comb)[0]
                        for each_nrf in range(0, pair_index.shape[0]):
                            selected.add(int(tf.floor(pair_index[each_nrf] / G_copy.shape[2])))
                    else:
                        output[n] = prev_out[n]
                        pair_index = np.nonzero(prev_out[n])[0]
                        for each_nrf in range(0, pair_index.shape[0]):
                            selected.add(int(tf.floor(pair_index[each_nrf] / G_copy.shape[2])))
        output = tf.constant(output, dtype=tf.float32)
        return output

    return model
def partial_feedback_semi_exhaustive_model(N_rf, B, p, M, K, sigma2):
    # uniformly quantize the values then pick the top Nrf to output
    def model(G):
        G = (tf.abs(G))
        top_values, top_indices = tf.math.top_k(G, k=p)
        temp = tf.keras.layers.Reshape((K * p,))(top_values)
        # min = tf.tile(tf.expand_dims(tf.reduce_min(temp, axis=1), axis=[1,2]), (1, K, p))
        min = tf.tile(tf.keras.layers.Reshape((1, 1))(tf.reduce_min(temp, axis=1)), (1, K, p))
        max = tf.tile(tf.keras.layers.Reshape((1, 1))(tf.reduce_max(temp, axis=1)), (1, K, p))
        top_values_quantized = tf.round((top_values - min) / (max - min) * (2 ** B - 1)) / (2 ** B - 1)
        # top_values_quantized = top_values
        return sparse_greedy_hueristic(N_rf, sigma2, K, M, p)(top_values_quantized, top_indices)

    return model
def partial_feedback_pure_greedy_model(N_rf, B, p, M, K, sigma2):
    # uniformly quantize the values then pick the top Nrf to output
    def model(G):
        G = (tf.abs(G))
        p=2
        top_values, top_indices = tf.math.top_k(G, k=p)
        return sparse_pure_greedy_hueristic(N_rf, sigma2, K, M, p)(top_values, top_indices, G)
    return model
def partial_feedback_pure_greedy_model_not_perfect_CSI_available(N_rf, B, p, M, K, sigma2):
    # uniformly quantize the values then pick the top Nrf to output
    def model(G):
        G = (tf.abs(G))
        top_values, top_indices = tf.math.top_k(G, k=p)
        G_copy = np.zeros((top_indices.shape[0], K, M))
        for n in range(0, top_indices.shape[0]):
            for i in range(0, K * p):
                # print(K*p)
                p_i = int(i % p)
                user_i = int(tf.floor(i / p))
                G_copy[n, user_i, int(top_indices[n, user_i, p_i])] = top_values[n, user_i, p_i]
        G_copy = tf.constant(G_copy, dtype=tf.float32)
        if p > 10:
            top_values, top_indices = tf.math.top_k(G, k=10)
            return sparse_pure_greedy_hueristic(N_rf, sigma2, K, M, 10)(top_values, top_indices, G_copy)
        return sparse_pure_greedy_hueristic(N_rf, sigma2, K, M, p)(top_values, top_indices, G_copy)
    return model
def partial_feedback_top_N_rf_model(N_rf, B, p, M, K, sigma2):
    # uniformly quantize the values then pick the top Nrf to output
    def model(G):
        G = tf.abs(G)
        top_values, top_indices, = tf.math.top_k(G, k=p)
        print(top_values.shape)
        temp = tf.keras.layers.Reshape((K * p,))(top_values)
        min = tf.tile(tf.keras.layers.Reshape((1, 1))(tf.reduce_min(temp, axis=1)), (1, K, p))
        max = tf.tile(tf.keras.layers.Reshape((1, 1))(tf.reduce_max(temp, axis=1)), (1, K, p))
        top_values_quantized = tf.round((top_values - min) / (max - min) * (2 ** B - 1)) / (2 ** B - 1) + 0.1
        G_prime = np.zeros(G.shape)
        for n in range(top_values_quantized.shape[0]):
            for k in range(0, K):
                for each_p in range(0, p):
                    G_prime[n, k, top_indices[n, k, each_p]] = top_values_quantized[n, k, each_p]
        return top_N_rf_user_model(M, K, N_rf)(tf.constant(G_prime, dtype=tf.float32))

    return model
def DP_partial_feedback_semi_exhaustive_model(N_rf, B, p, M, K, sigma2):
    # uniformly quantize the values then pick the top Nrf to output
    def model(G):
        G = (tf.abs(G))
        top_values, top_indices = tf.math.top_k(G, k=p)
        temp = tf.keras.layers.Reshape((K * p,))(top_values)
        # min = tf.tile(tf.expand_dims(tf.reduce_min(temp, axis=1), axis=[1,2]), (1, K, p))
        min = tf.tile(tf.keras.layers.Reshape((1, 1))(tf.reduce_min(temp, axis=1)), (1, K, p))
        max = tf.tile(tf.keras.layers.Reshape((1, 1))(tf.reduce_max(temp, axis=1)), (1, K, p))
        top_values_quantized = tf.round((top_values - min) / (max - min) * (2 ** B - 1)) / (2 ** B - 1)
        # top_values_quantized = top_values
        out = []
        prev_out = None
        for i in range(2, N_rf+1):
            prev_out = DP_sparse_greedy_hueristic(i, sigma2, K, M, p, i-1, prev_out)(top_values_quantized, top_indices)
            # print(prev_out)
            out.append(prev_out)
        return out
    return model
def DP_partial_feedback_pure_greedy_model(N_rf, B, p, M, K, sigma2, perfect_CSI=False, pick_top=2):
    # uniformly quantize the values then pick the top Nrf to output
    def model(G):
        G = (tf.abs(G))
        top_values, top_indices = tf.math.top_k(G, k=p)
        if perfect_CSI == False:
            # quantization ===
            G, G_max = Input_normalization_per_user(G)
            G = tf.where(G > 1, 1, G)
            G = tf.round(G * (2 ** B - 1)) / (2 ** B - 1)
            G = tf.multiply(G, G_max)
            # quantization ===
            G_copy = np.zeros((top_indices.shape[0], K, M))
            for n in range(0, top_indices.shape[0]):
                for i in range(0, K * p):
                    # print(K*p)
                    p_i = int(i % p)
                    user_i = int(tf.floor(i / p))
                    G_copy[n, user_i, int(top_indices[n, user_i, p_i])] = top_values[n, user_i, p_i]
            G_copy = tf.constant(G_copy, dtype=tf.float32)
            G = G_copy
        if p > pick_top:
            top_values, top_indices = tf.math.top_k(G, k=pick_top)
        # top_values_quantized = top_values
        out = []
        prev_out = None
        G_original = G
        for i in range(1, N_rf+1):
            G = G_original/tf.sqrt(i * 1.0)
            pp = min(p, pick_top)
            prev_out = DP_sparse_pure_greedy_hueristic(i, sigma2, K, M, pp, G, i-1, prev_out)(top_values, top_indices)
            # print(prev_out)
            out.append(prev_out)
        return out
    return model
def DP_DNN_feedback_pure_greedy_model_past(N_rf, p, M, K, sigma2, encoder_decoder, perfect_CSI=False, pick_top=2):
    # uniformly quantize the values then pick the top Nrf to output
    feed_back_model = max_min_k_link_feedback_model(1, B, p, M, K)
    def model(G):
        # quantization ===
        valid_data = generate_link_channel_data(1000, K, M, Nrf=N_rf)
        garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
        q_train_data = tf.abs(G) / max_val
        q_train_data = tf.where(q_train_data > 1.0, 1.0, q_train_data)
        q_train_data = (tf.round(q_train_data * (2 ** 6 - 1)) / (2 ** 6 - 1) + 1 / (2 ** (6 + 1))) * max_val
        # quantization ===
        if not encoder_decoder is None:
            G = encoder_decoder(q_train_data)
        top_values, top_indices = tf.math.top_k(G, k=pick_top)
        # top_values_quantized = top_values
        out = []
        prev_out = None
        G_original = G
        for i in range(1, N_rf+1):
            G = G_original/tf.sqrt(i * 1.0)
            pp = min(p, pick_top)
            prev_out = DP_sparse_pure_greedy_hueristic(i, sigma2, K, M, pp, G, i-1, prev_out)(top_values, top_indices)
            # print(prev_out)
            out.append(prev_out)
        return out
    return model
def partial_feedback_pure_greedy_model_weighted_SR(N_rf, B, p, M, K, sigma2, environment):
    # uniformly quantize the values then pick the top Nrf to output
    def model(G):
        G = (tf.abs(G))
        p=2
        top_values, top_indices = tf.math.top_k(G, k=p)
        return sparse_pure_greedy_hueristic_weighted_SR(N_rf, sigma2, K, M, p, environment)(top_values, top_indices, G)
    return model
def sparse_pure_greedy_hueristic_weighted_SR(N_rf, sigma2, K, M, p, environment):
    def model(top_val, top_indice, G=None):
        output = np.zeros((top_indice.shape[0], K * M))
        if G is None:
            G_copy = np.zeros((top_indice.shape[0], K, M))
            for n in range(0, top_indice.shape[0]):
                for i in range(0, K * p):
                    # print(K*p)
                    p_i = int(i % p)
                    user_i = int(tf.floor(i / p))
                    G_copy[n, user_i, int(top_indice[n, user_i, p_i])] = top_val[n, user_i, p_i]
            G_copy = tf.constant(G_copy, dtype=tf.float32)
            G = G_copy
        print("done generating partial information")
        for n in range(0, G.shape[0]):
            # print("==================================== type", n, "====================================")
            combinations = []
            for index_1 in range(0, K * p):
                p_1 = int(index_1 % p)
                user_1 = int(tf.floor(index_1 / p))
                comb = np.zeros((K * M,))
                comb[user_1 * M + top_indice[n, user_1, p_1]] = 1
                combinations.append(comb)
            min = 100
            best_one = None
            for com in combinations:
                current_min = tf.reduce_mean(environment.compute_weighted_loss(tf.expand_dims(tf.constant(com, tf.float32), 0), G[n:n + 1], update=False))
                if current_min < min:
                    min = current_min
                    best_one = com
            output[n] = best_one
            selected = set()
            pair_index = np.nonzero(best_one)
            selected.add(int(tf.floor(pair_index[0][0] / G.shape[2])))
            if N_rf >= 2:
                for n_rf in range(1, N_rf):
                    new_comb = []
                    for additional_i in range(0, K * p):
                        p_i = int(additional_i % p)
                        user_i = int(tf.floor(additional_i / p))
                        beamformer_i = int(top_indice[n, user_i, p_i])
                        if output[n, user_i * M + beamformer_i] != 1:
                            if not int(tf.floor((user_i * M + beamformer_i) / M)) in selected:
                                temp = output[n].copy()
                                temp[user_i * M + beamformer_i] = 1
                                new_comb.append(temp)
                    min = 100
                    best_comb = None
                    for com in new_comb:
                        current_min = tf.reduce_mean(environment.compute_weighted_loss(tf.expand_dims(tf.constant(com, tf.float32), 0), G[n:n + 1], update=False))
                        if current_min < min:
                            min = current_min
                            best_comb = com
                    output[n] = best_comb
                    pair_index = np.nonzero(best_comb)[0]
                    for each_nrf in range(0, pair_index.shape[0]):
                        selected.add(int(tf.floor(pair_index[each_nrf] / G.shape[2])))
        output = tf.constant(output, dtype=tf.float32)
        return output

    return model


############################## Misc Models ##############################
############################## Layers ##############################
def relaxation_based_solver(M, K, N_rf, sigma=1.0):
    def solver(G):
        G = tf.abs(G)
        g_top = tf.tile(tf.reduce_max(G, axis=1, keepdims=True), (1, K, 1))
        g_second_top = tf.tile(tf.reduce_max(tf.where(G >= g_top, 0.0, G), axis=1, keepdims=True), (1, K, 1))
        g_max = tf.where(G >= g_top, g_second_top, g_top)
        score = tf.math.log(1.0 + tf.divide(G, g_max * (N_rf-1) + sigma))
        score = tf.keras.layers.Reshape((K*M,))(score)
        decision = Harden_scheduling_user_constrained(N_rf, K, M, default_val=0)(score)
        return decision
    return solver
def k_link_feedback_model(N_rf, B, p, M, K, g_max):
    def model(G, g_max=g_max):
        G = (tf.abs(G))
        top_values, top_indices = tf.math.top_k(G, k=p)
        G_copy = np.zeros((top_indices.shape[0], K, M))
        for n in range(0, top_indices.shape[0]):
            for i in range(0, K * p):
                p_i = int(i % p)
                user_i = int(tf.floor(i / p))
                G_copy[n, user_i, int(top_indices[n, user_i, p_i])] = top_values[n, user_i, p_i]
        G_copy = tf.constant(G_copy, dtype=tf.float32)
        G = G_copy
        G = tf.divide(G, g_max)
        G = tf.where(G > 1, 1, G)
        G = tf.round(G * (2 ** B - 1)) / (2 ** B - 1)
        G = tf.multiply(G, g_max)
        return G
    return model

def DP_DNN_feedback_pure_greedy_model(N_rf, B, p, M, K, sigma2, perfect_CSI=False, pick_top=2):
    # uniformly quantize the values then pick the top Nrf to output
    feed_back_model = max_min_k_link_feedback_model(1, B, p, M, K)
    def model(G):
        G = (tf.abs(G))
        # quantization ===
        if perfect_CSI == False:
            G = feed_back_model(G)
        # quantization ===
        top_values, top_indices = tf.math.top_k(G, k=p)
        if p > pick_top:
            top_values, top_indices = tf.math.top_k(G, k=pick_top)
        # top_values_quantized = top_values
        out = []
        prev_out = None
        G_original = G
        for i in range(1, N_rf+1):
            G = G_original/tf.sqrt(i * 1.0)
            pp = min(p, pick_top)
            prev_out = DP_sparse_pure_greedy_hueristic(i, sigma2, K, M, pp, G, i-1, prev_out)(top_values, top_indices)
            # print(prev_out)
            out.append(prev_out)
        return out
    return model

def max_min_k_link_feedback_model(N_rf, B, p, M, K):
    init_ds = generate_link_channel_data(1000, K, M, N_rf)
    G = (tf.abs(init_ds))
    top_values, top_indices = tf.math.top_k(G, k=p)
    G_copy = np.zeros((top_indices.shape[0], K, M))
    for n in range(0, top_indices.shape[0]):
        for i in range(0, K * p):
            p_i = int(i % p)
            user_i = int(tf.floor(i / p))
            G_copy[n, user_i, int(top_indices[n, user_i, p_i])] = top_values[n, user_i, p_i]
    G_topk = G_copy
    g_max = tf.reshape(G_topk, (G_topk.shape[0], M * K))
    g_max = tf.reduce_max(g_max, axis=1)
    g_max = tf.reduce_mean(g_max)
    g_min = tf.reshape(G_topk, (G_topk.shape[0], M * K))
    g_min = tf.reduce_mean(tf.sort(g_min, axis=1)[:, (M - p) * K])
    g_min = tf.cast(g_min, tf.float32)
    g_max = tf.cast(g_max, tf.float32)
    def q_4(G, B=B, g_max=g_max, g_min=g_min): # limited range between max and min
        G = (tf.abs(G))
        top_values, top_indices = tf.math.top_k(G, k=p)
        G_copy = np.zeros((top_indices.shape[0], K, M))
        for n in range(0, top_indices.shape[0]):
            for i in range(0, K * p):
                p_i = int(i % p)
                user_i = int(tf.floor(i / p))
                G_copy[n, user_i, int(top_indices[n, user_i, p_i])] = top_values[n, user_i, p_i]
        G_topk = G_copy
        mask = tf.where(G_topk != 0, 1.0, 0.0)
        G_temp = tf.divide(G_topk-mask*g_min, (g_max-g_min))
        G_temp = tf.where(G_temp > 1, 1, G_temp)
        G_temp = tf.round(G_temp * (2 ** B - 1)) / (2 ** B - 1)
        G_temp = tf.multiply(G_temp, (g_max-g_min)) + g_min * mask
        return G_temp
    return q_4
class iterative_NN_scheduler():
    def __init__(self, model, iteration, loss1, lr, loss2=None):
        self.model = model
        self.iter = iteration
        self.loss1 = loss1
        self.loss2 = loss2
        self.optimizer = tf.keras.optimizers.Adam(lr=lr)
    def __call__(self, features, *args, **kwargs):
        return self.model(features)
    def call_with_gradient(self, features):
        scheduled_output = None
        raw_output = None
        for iter in range(self.iter):
            with tf.GradientTape(persistent=True) as tape:
                scheduled_output, raw_output = self.model(features)
                loss_1 = 0
                loss_4 = 0
                for i in range(0, scheduled_output.shape[1]):
                    sr = self.loss1(scheduled_output[:, i], features)
                    loss_1 = loss_1 + tf.exp(
                        tf.constant(-scheduled_output.shape[1] + 1 + i, dtype=tf.float32)) * sr
                    # ce = All_softmaxes_MSE_general(N_rf, K, M)(raw_output[:, i])
                    # ce = All_softmaxes_CE_general(N_rf, K, M)(raw_output[:, i])
                    # loss_4 = loss_4 + 0.1 * tf.exp(tf.constant(-scheduled_output.shape[1]+1+i, dtype=tf.float32)) * ce
                    # mask = partial_feedback_pure_greedy_model(N_rf, 32, 10, M, K, sigma2_n)(features)

                    mask = tf.stop_gradient(
                        Harden_scheduling_user_constrained(1, K, M, default_val=0)(scheduled_output[:, i]))
                    ce = tf.keras.losses.CategoricalCrossentropy()(scheduled_output[:, i] / N_rf, mask / N_rf)
                    loss_4 = loss_4 + 0.1 * tf.exp(tf.constant(-scheduled_output.shape[1] + 1 + i, dtype=tf.float32)) * ce
                # # print("==============================")
                loss = loss_1 + loss_4
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            del tape
        return scheduled_output, raw_output

############################## Custom Layers ##############################

class Closest_embedding_layer(tf.keras.layers.Layer):
    def __init__(self, user_count=2, embedding_count=8, bit_count=15, i=0, **kwargs):
        super(Closest_embedding_layer, self).__init__()
        self.user_count = user_count
        self.bit_count = bit_count
        self.embedding_count = embedding_count
        initializer = tf.keras.initializers.he_uniform()
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
        # E is in the shape of [E, 2**B]
        self.E = self.add_weight(name='embedding{}'.format(i),
                                 shape=(self.bit_count, self.embedding_count),
                                 initializer=initializer,
                                 trainable=True)
        self.i = i
    def call(self, z, training=True):
        # z = z + tf.random.normal([z.shape[1], z.shape[2]], 0, tf.math.reduce_std(z, axis=0)/10.0)
        # z is in the shape of [None, K, E]
        # print(tf.keras.sum(z**2, axis=1, keepdims=True).shape)
        distances = (KB.sum(z ** 2, axis=2, keepdims=True)
                     - 2 * KB.dot(z, self.E)
                     + KB.sum(self.E ** 2, axis=0, keepdims=True))
        encoding_indices = KB.argmax(-distances, axis=2)
        # print(encoding_indices)
        # encodings = tf.gather(tf.transpose(self.E), encoding_indices)
        encodings = tf.nn.embedding_lookup(tf.transpose(self.E), encoding_indices)
        # if not (encoding_indices.shape[0] is None):
        #     print(np.unique(encoding_indices.numpy()).shape)
        return encodings
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'bit_count': self.bit_count,
            'user_count': self.user_count,
            'embedding_count': self.embedding_count,
            'i': self.i,
            'name': "Closest_embedding_layer_{}".format(self.i)
        })
        return config
class Closest_embedding_layer_moving_avg(tf.keras.layers.Layer):
    def __init__(self, user_count=2, embedding_count=8, bit_count=15, i=0, **kwargs):
        super(Closest_embedding_layer_moving_avg, self).__init__(name="Closest_embedding_layer_moving_avg")
        self.user_count = user_count
        self.bit_count = bit_count
        self.embedding_count = embedding_count
        initializer = tf.keras.initializers.he_uniform()
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
        # E is in the shape of [E, 2**B]
        self.E = self.add_weight(name='embedding',
                                 shape=(self.bit_count, self.embedding_count),
                                 initializer=initializer,
                                 trainable=False)
        self.i = i
        self.last_m = self.add_weight(name='last_m',
                                      shape=(self.bit_count, self.embedding_count),
                                      initializer=initializer,
                                      trainable=False)

    def call(self, z, training=True):
        # if training:
        #     z = z + tf.random.normal([z.shape[1], z.shape[2]], 0, tf.math.reduce_std(z, axis=0))
        # z is in the shape of [None, K, E]
        # print(tf.keras.sum(z**2, axis=1, keepdims=True).shape)
        distances = (KB.sum(z ** 2, axis=2, keepdims=True)
                     - 2 * KB.dot(z, self.E)
                     + KB.sum(self.E ** 2, axis=0, keepdims=True))
        encoding_indices = KB.argmax(-distances, axis=2)
        # encodings = tf.gather(tf.transpose(self.E), encoding_indices)
        encodings = tf.nn.embedding_lookup(tf.transpose(self.E), encoding_indices)
        if not (encoding_indices.shape[0] is None):
            print(np.unique(encoding_indices.numpy()))
        return encodings

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'bit_count': self.bit_count,
            'user_count': self.user_count,
            'embedding_count': self.embedding_count,
            'i': self.i,
            'name': self.name
        })
        return config
class Interference_Input_modification(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Interference_Input_modification, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)

    def call(self, x, input_mod, step):
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
        power = tf.tile(tf.expand_dims(tf.reduce_sum( input_mod, axis=1), 1), (1, self.K, 1)) - input_mod
        interference_f = tf.multiply(power, x)
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        interference_t = tf.matmul(input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
        interference_t = input_reshaper(interference_t)
        interference_f = input_reshaper(interference_f)
        x = tf.keras.layers.Reshape((self.K * self.M,))(x)
        # x = tf.reduce_sum(x, axis=1, keepdims=True)
        x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), interference_t) + tf.constant(step))
        input_i = input_concatnator(
            [input_reshaper(input_mod), interference_t, interference_f, x, iteration_num])
        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Interference_Input_modification"
        })
        return config
class Interference_Input_modification_no_loop(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Interference_Input_modification_no_loop, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)

    def call(self, x, input_mod):
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
        power = tf.tile(tf.expand_dims(tf.reduce_sum(input_mod, axis=1), 1), (1, self.K, 1)) - input_mod
        interference_f = tf.multiply(power, x)
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        interference_t = tf.matmul(input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
        interference_t = input_reshaper(interference_t)
        interference_f = input_reshaper(interference_f)
        input_i = input_concatnator(
            [input_reshaper(input_mod), interference_t, interference_f])
        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Interference_Input_modification_no_loop"
        })
        return config
class Interference_Input_modification_per_user(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Interference_Input_modification_per_user, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)

    def call(self, x, input_mod, step):
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.K, self.M))
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        interference_t = tf.matmul(input_mod, unflattened_output_0)
        interference_f = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=1)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, 1))
        interference_f = tf.tile(tf.expand_dims(interference_f, 2), (1, 1, 1))
        x = tf.keras.layers.Reshape((self.K * self.M,))(x)
        x = tf.expand_dims(x, axis=1)
        x = tf.tile(x, (1, self.K, 1))
        iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), interference_t) + tf.constant(step))
        input_i = input_concatnator(
            [input_reshaper(input_mod), interference_t, interference_f, x, iteration_num])
        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Interference_Input_modification_per_user"
        })
        return config
class Per_link_Input_modification_more_G_less_X(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Per_link_Input_modification_more_G_less_X, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, x, input_mod, step):
        if self.Mk is None:
            self.Mk = np.zeros((self.K*self.M, self.K), dtype=np.float32)
            self.Mm = np.zeros((self.K*self.M, self.M), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.M):
                    self.Mk[i*self.M+j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.K):
                    self.Mm[i*self.K+j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
        power = tf.tile(tf.expand_dims(tf.reduce_sum(input_mod, axis=1), 1), (1, self.K, 1)) - input_mod
        interference_f = tf.multiply(power, x)
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        interference_t = tf.matmul(input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
        interference_t = input_reshaper(interference_t)
        interference_f = input_reshaper(interference_f)
        G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M*self.K, ))(input_mod), axis=1, keepdims=True)
        G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, self.K * self.M, 1))
        G_user_mean = tf.reduce_mean(input_mod, axis=2, keepdims=True)
        G_user_mean = tf.matmul(self.Mk, G_user_mean)
        affected = tf.matmul(self.Mk, input_mod)
        effectedBy = tf.matmul(self.Mm, tf.transpose(input_mod, (0, 2, 1)))
        # x = tf.reduce_sum(x, axis=2)
        x = tf.keras.layers.Reshape((self.K*self.M, ))(x)
        # x = tf.reduce_sum(x, axis=1, keepdims=True)
        x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)) + tf.constant(step))

        input_i = input_concatnator(
            [input_reshaper(input_mod), G_user_mean, G_mean, interference_t, interference_f, x, iteration_num])
        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Per_link_Input_modification_more_G_less_X",
            'Mk': None,
            'Mm': None
        })
        return config
class Per_link_Input_modification_more_G(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Per_link_Input_modification_more_G, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, x, input_mod, step):
        if self.Mk is None:
            self.Mk = np.zeros((self.K*self.M, self.K), dtype=np.float32)
            self.Mm = np.zeros((self.K*self.M, self.M), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.M):
                    self.Mk[i*self.M+j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.K):
                    self.Mm[i*self.K+j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))

        power = tf.tile(tf.expand_dims(tf.reduce_sum(input_mod, axis=1), 1), (1, self.K, 1)) - input_mod
        interference_f = tf.multiply(power, x)
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        interference_t = tf.matmul(input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
        interference_t = input_reshaper(interference_t)
        interference_f = input_reshaper(interference_f)

        affected = tf.matmul(self.Mk, input_mod)
        effectedBy = tf.matmul(self.Mm, tf.transpose(input_mod, (0, 2, 1)))
        x = tf.keras.layers.Reshape((self.K*self.M, ))(x)
        # x = tf.reduce_sum(x, axis=2)
        # x = tf.reduce_sum(x, axis=1, keepdims=True)
        x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)) + tf.constant(step))
        input_i = input_concatnator(
            [input_reshaper(input_mod), affected, effectedBy, interference_t, interference_f, x, iteration_num])
        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Per_link_Input_modification_more_G",
            'Mk': None,
            'Mm': None
        })
        return config
class Per_link_Input_modification_most_G(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Per_link_Input_modification_most_G, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, x, input_mod, step):
        step = int(step)
        if self.Mk is None:
            self.Mk = np.zeros((self.K*self.M, self.K), dtype=np.float32)
            self.Mm = np.zeros((self.K*self.M, self.M), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.M):
                    self.Mk[i*self.M+j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.K):
                    self.Mm[i*self.K+j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
        # up = tf.multiply(input_mod, x)
        # interference_f = tf.tile(tf.reduce_sum(up, axis=1, keepdims=True), (1, self.K, 1)) - up
        power = tf.tile(tf.reduce_sum(input_mod, axis=1, keepdims=True), (1, self.K, 1)) - input_mod
        interference_f = tf.multiply(power, x)
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        interference_t = tf.matmul(input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
        interference_t = input_reshaper(interference_t)
        interference_f = input_reshaper(interference_f)
        G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M*self.K, ))(input_mod), axis=1, keepdims=True)
        G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, self.K * self.M, 1))
        G_max = tf.reduce_max(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
        G_max = tf.tile(tf.expand_dims(G_max, axis=1), (1, self.K * self.M, 1))
        G_min = tf.reduce_min(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
        G_min = tf.tile(tf.expand_dims(G_min, axis=1), (1, self.K * self.M, 1))
        G_user_mean = tf.reduce_mean(input_mod, axis=2, keepdims=True)
        G_user_mean = tf.matmul(self.Mk, G_user_mean)
        G_user_max = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_max = tf.matmul(self.Mk, G_user_max)
        G_user_min = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_min = tf.matmul(self.Mk, G_user_min)
        G_col_mean = tf.transpose(tf.reduce_mean(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        G_col_mean = tf.matmul(self.Mm, G_col_mean)
        G_col_max = tf.transpose(tf.reduce_max(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        G_col_max = tf.matmul(self.Mm, G_col_max)
        G_col_min = tf.transpose(tf.reduce_max(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        G_col_min = tf.matmul(self.Mm, G_col_min)
        # G_col = tf.matmul(self.Mk, input_mod)
        # x = tf.reduce_sum(x, axis=2)
        x = tf.keras.layers.Reshape((self.K*self.M, ))(x)
        # x = tf.reduce_sum(x, axis=1, keepdims=True)
        x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)))
        # print(iteration_num.shape)
        num = np.zeros((1, self.k))
        num[0, step] = 1
        iteration_num = tf.tile(iteration_num, (1, 1, self.k)) + num
        input_i = input_concatnator(
            [-input_reshaper(input_mod),
             G_mean, G_max, G_min,
             # G_mean,
             G_user_mean, G_user_min, G_user_max,
             G_col_max, G_col_min, G_col_mean,
             interference_t, interference_f,
             x,
             iteration_num])
        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Per_link_Input_modification_most_G",
            'Mk': None,
            'Mm': None
        })
        return config
class Per_link_Input_modification_most_G_col(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Per_link_Input_modification_most_G_col, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, x, input_mod, step):
        if self.Mk is None:
            self.Mk = np.zeros((self.K*self.M, self.K), dtype=np.float32)
            self.Mm = np.zeros((self.K*self.M, self.M), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.M):
                    self.Mk[i*self.M+j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.K):
                    self.Mm[i*self.K+j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
        # up = tf.multiply(input_mod, x)
        # interference_f = tf.tile(tf.reduce_sum(up, axis=1, keepdims=True), (1, self.K, 1)) - up
        power = tf.tile(tf.reduce_sum(input_mod, axis=1, keepdims=True), (1, self.K, 1)) - input_mod
        interference_f = tf.multiply(power, x)
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        interference_t = tf.matmul(input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
        interference_t = input_reshaper(interference_t)
        interference_f = input_reshaper(interference_f)
        G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M*self.K, ))(input_mod), axis=1, keepdims=True)
        G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, self.K * self.M, 1))
        G_max = tf.reduce_max(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
        G_max = tf.tile(tf.expand_dims(G_max, axis=1), (1, self.K * self.M, 1))
        G_min = tf.reduce_min(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
        G_min = tf.tile(tf.expand_dims(G_min, axis=1), (1, self.K * self.M, 1))
        G_user_mean = tf.reduce_mean(input_mod, axis=2, keepdims=True)
        G_user_mean = tf.matmul(self.Mk, G_user_mean)
        G_user_max = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_max = tf.matmul(self.Mk, G_user_max)
        G_user_min = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_min = tf.matmul(self.Mk, G_user_min)
        # G_col_mean = tf.transpose(tf.reduce_mean(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        # G_col_mean = tf.matmul(self.Mm, G_col_mean)
        # G_col_max = tf.transpose(tf.reduce_max(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        # G_col_max = tf.matmul(self.Mm, G_col_max)
        # G_col_min = tf.transpose(tf.reduce_max(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        # G_col_min = tf.matmul(self.Mm, G_col_min)
        G_col = tf.matmul(self.Mm, tf.transpose(input_mod, perm=[0,2,1]))
        # G_col = tf.matmul(self.Mk, input_mod)
        # x = tf.reduce_sum(x, axis=2)
        x = tf.keras.layers.Reshape((self.K*self.M, ))(x)
        # x = tf.reduce_sum(x, axis=1, keepdims=True)
        x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)) + tf.constant(step))

        input_i = input_concatnator(
            [-input_reshaper(input_mod),
             G_mean, G_max, G_min,
             # G_mean,
             G_user_mean, G_user_min, G_user_max,
             G_col,
             interference_t, interference_f,
             x,
             iteration_num])
        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Per_link_Input_modification_most_G",
            'Mk': None,
            'Mm': None
        })
        return config
class Per_link_Input_modification_most_G_col_lessX(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Per_link_Input_modification_most_G_col_lessX, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, x, input_mod, step):
        if self.Mk is None:
            self.Mk = np.zeros((self.K*self.M, self.K), dtype=np.float32)
            self.Mm = np.zeros((self.K*self.M, self.M), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.M):
                    self.Mk[i*self.M+j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.K):
                    self.Mm[i*self.K+j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
        up = tf.multiply(input_mod, x)
        interference_f = tf.tile(tf.reduce_sum(up, axis=1, keepdims=True), (1, self.K, 1)) - up
        # power = tf.tile(tf.reduce_sum(input_mod, axis=1, keepdims=True), (1, self.K, 1)) - input_mod
        # interference_f = tf.multiply(power, x)
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        interference_t = tf.matmul(input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
        interference_t = input_reshaper(interference_t)
        interference_f = input_reshaper(interference_f)
        G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M*self.K, ))(input_mod), axis=1, keepdims=True)
        G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, self.K * self.M, 1))
        G_max = tf.reduce_max(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
        G_max = tf.tile(tf.expand_dims(G_max, axis=1), (1, self.K * self.M, 1))
        G_min = tf.reduce_min(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
        G_min = tf.tile(tf.expand_dims(G_min, axis=1), (1, self.K * self.M, 1))
        G_user_mean = tf.reduce_mean(input_mod, axis=2, keepdims=True)
        G_user_mean = tf.matmul(self.Mk, G_user_mean)
        G_user_max = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_max = tf.matmul(self.Mk, G_user_max)
        G_user_min = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_min = tf.matmul(self.Mk, G_user_min)
        # G_col_mean = tf.transpose(tf.reduce_mean(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        # G_col_mean = tf.matmul(self.Mm, G_col_mean)
        # G_col_max = tf.transpose(tf.reduce_max(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        # G_col_max = tf.matmul(self.Mm, G_col_max)
        # G_col_min = tf.transpose(tf.reduce_max(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        # G_col_min = tf.matmul(self.Mm, G_col_min)
        G_col = tf.matmul(self.Mm, tf.transpose(input_mod, perm=[0,2,1]))
        # G_col = tf.matmul(self.Mk, input_mod)
        # x = tf.reduce_sum(x, axis=2)
        # x = tf.keras.layers.Reshape((self.K*self.M, ))(x)
        x_temp = tf.reduce_sum(x, axis=2, keepdims=True)
        x_current_user = tf.matmul(self.Mk, x_temp)
        x_temp = tf.transpose(x_temp, perm = [0,2,1])
        x = tf.tile(x_temp, [1,self.K * self.M, 1])

        # x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        tim = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)) + 1)[:, ]
        iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)) + step)[:, ]
        num = tf.matmul(self.Mk, tf.ones((self.K, self.K), dtype=tf.float32))
        num = tf.multiply(tf.tile(tim, [1, 1, self.K]), num)
        input_i = input_concatnator(
            [-input_reshaper(input_mod), -input_reshaper(input_mod), -input_reshaper(input_mod),
             G_mean, G_max, G_min,
             # G_mean,
             G_user_mean, G_user_min, G_user_max,
             G_col,
             interference_t, interference_f,
             x, x_current_user, num
             # iteration_num
             ])
        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Per_link_Input_modification_most_G_col_lessX",
            'Mk': None,
            'Mm': None
        })
        return config
class Per_link_Input_modification_most_G_no_x(tf.keras.layers.Layer):
        def __init__(self, K, M, N_rf, k, **kwargs):
            super(Per_link_Input_modification_most_G_no_x, self).__init__()
            self.K = K
            self.M = M
            self.N_rf = N_rf
            self.k = k
            self.Mk = None
            self.Mm = None
            # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)

        def call(self, x, input_mod, step):
            if self.Mk is None:
                self.Mk = np.zeros((self.K * self.M, self.K), dtype=np.float32)
                self.Mm = np.zeros((self.K * self.M, self.M), dtype=np.float32)
                for i in range(0, self.K):
                    for j in range(0, self.M):
                        self.Mk[i * self.M + j, i] = 1.0
                for i in range(0, self.M):
                    for j in range(0, self.K):
                        self.Mm[i * self.K + j, i] = 1.0
                # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
                # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)
            input_concatnator = tf.keras.layers.Concatenate(axis=2)
            input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
            up = tf.multiply(input_mod, x)
            interference_f = tf.tile(tf.reduce_sum(up, axis=1, keepdims=True), (1, self.K, 1)) - up
            # power = tf.tile(tf.reduce_sum(input_mod, axis=1, keepdims=True), (1, self.K, 1)) - input_mod
            # interference_f = tf.multiply(power, x)
            unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
            interference_t = tf.matmul(input_mod, unflattened_output_0)
            interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
            interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
            interference_t = input_reshaper(interference_t)
            interference_f = input_reshaper(interference_f)
            G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
            G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, self.K * self.M, 1))
            G_max = tf.reduce_max(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
            G_max = tf.tile(tf.expand_dims(G_max, axis=1), (1, self.K * self.M, 1))
            G_min = tf.reduce_min(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
            G_min = tf.tile(tf.expand_dims(G_min, axis=1), (1, self.K * self.M, 1))
            G_user_mean = tf.reduce_mean(input_mod, axis=2, keepdims=True)
            G_user_mean = tf.matmul(self.Mk, G_user_mean)
            G_user_max = tf.reduce_max(input_mod, axis=2, keepdims=True)
            G_user_max = tf.matmul(self.Mk, G_user_max)
            G_user_min = tf.reduce_max(input_mod, axis=2, keepdims=True)
            G_user_min = tf.matmul(self.Mk, G_user_min)
            G_col = tf.matmul(self.Mm, tf.transpose(input_mod, perm=[0, 2, 1]))
            input_i = input_concatnator(
                [-input_reshaper(input_mod),
                 G_mean, G_max, G_min,
                 # G_mean,
                 G_user_mean, G_user_min, G_user_max,
                 G_col,
                 interference_t, interference_f])
            return input_i

        def get_config(self):
            config = super().get_config().copy()
            config.update({
                'K': self.K,
                'M': self.M,
                'N_rf': self.N_rf,
                'k': self.k,
                'name': "Per_link_Input_modification_most_G_no_x",
                'Mk': None,
                'Mm': None
            })
            return config
class Per_link_Input_modification_most_G_sigmoid(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Per_link_Input_modification_most_G_sigmoid, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, x, input_mod, step):
        if self.Mk is None:
            self.Mk = np.zeros((self.K*self.M, self.K), dtype=np.float32)
            self.Mm = np.zeros((self.K*self.M, self.M), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.M):
                    self.Mk[i*self.M+j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.K):
                    self.Mm[i*self.K+j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)

        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
        power = tf.tile(tf.expand_dims(tf.reduce_sum(input_mod, axis=1), 1), (1, self.K, 1)) - input_mod
        interference_f = tf.multiply(power, x)
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        interference_t = tf.matmul(input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
        interference_t = input_reshaper(interference_t)
        interference_f = input_reshaper(interference_f)
        input_mod = input_mod * 1.0 - x
        G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M*self.K, ))(input_mod), axis=1, keepdims=True)
        G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, self.K * self.M, 1))
        G_max = tf.reduce_max(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
        G_max = tf.tile(tf.expand_dims(G_max, axis=1), (1, self.K * self.M, 1))
        G_min = tf.reduce_min(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
        G_min = tf.tile(tf.expand_dims(G_min, axis=1), (1, self.K * self.M, 1))
        G_user_mean = tf.reduce_mean(input_mod, axis=2, keepdims=True)
        G_user_mean = tf.matmul(self.Mk, G_user_mean)
        G_user_max = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_max = tf.matmul(self.Mk, G_user_max)
        G_user_min = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_min = tf.matmul(self.Mk, G_user_min)
        G_col_mean = tf.transpose(tf.reduce_mean(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        G_col_mean = tf.matmul(self.Mm, G_col_mean)
        G_col_max = tf.transpose(tf.reduce_max(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        G_col_max = tf.matmul(self.Mm, G_col_max)
        G_col_min = tf.transpose(tf.reduce_max(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        G_col_min = tf.matmul(self.Mm, G_col_min)
        # x = tf.reduce_sum(x, axis=2)
        x = tf.keras.layers.Reshape((self.K*self.M, ))(x)
        # x = tf.reduce_sum(x, axis=1, keepdims=True)
        x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)) + tf.constant(step))
        input_i = input_concatnator(
            [input_reshaper(input_mod),
             G_mean, G_max, G_min,
             # G_mean,
             G_user_mean, G_user_min, G_user_max,
             G_col_max, G_col_min, G_col_mean,
             interference_t, interference_f,
             x,
             iteration_num])
        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Per_link_Input_modification_most_G_sigmoid",
            'Mk': None,
            'Mm': None
        })
        return config
class Sequential_Per_link_Input_modification_most_G_raw_self(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Sequential_Per_link_Input_modification_most_G_raw_self, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, x, input_mod, step):
        if self.Mk is None:
            self.Mk = np.zeros((self.K*self.M, self.K), dtype=np.float32)
            self.Mm = np.zeros((self.K*self.M, self.M), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.M):
                    self.Mk[i*self.M+j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.K):
                    self.Mm[i*self.K+j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)
        # x = tf.reduce_sum(x_raw, axis=2)
        x = tf.keras.layers.Reshape((self.K, self.M))(x)
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
        power = tf.tile(tf.expand_dims(tf.reduce_sum(input_mod, axis=1), 1), (1, self.K, 1)) - input_mod
        interference_f = tf.multiply(power, x)
        up = tf.multiply(input_mod, x)
        interference_f_2 = tf.tile(tf.reduce_sum(up, axis=1, keepdims=True), (1, self.K, 1)) - up
        selected = tf.keras.layers.Reshape((self.M*self.K, 1))(tf.multiply(x, input_mod))
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        interference_t = tf.matmul(input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
        interference_t = input_reshaper(interference_t)
        interference_f = input_reshaper(interference_f)
        interference_f_2 = input_reshaper(interference_f_2)
        G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M*self.K, ))(input_mod), axis=1, keepdims=True)
        G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, self.K * self.M, 1))
        G_max = tf.reduce_max(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
        G_max = tf.tile(tf.expand_dims(G_max, axis=1), (1, self.K * self.M, 1))
        G_min = tf.reduce_min(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
        G_min = tf.tile(tf.expand_dims(G_min, axis=1), (1, self.K * self.M, 1))
        G_user_tiled = tf.matmul(self.Mk, input_mod)
        G_user_tiled = tf.multiply(tf.tile(1.0-tf.eye(self.M), (self.K, 1)), G_user_tiled)
        G_user_mean = tf.reduce_mean(G_user_tiled, axis=2, keepdims=True)
        G_user_max = tf.reduce_max(G_user_tiled, axis=2, keepdims=True)
        G_user_min = tf.reduce_min(G_user_tiled, axis=2, keepdims=True)
        GX_user_mean = tf.reduce_mean(tf.multiply(input_mod, x), axis=2, keepdims=True)
        GX_user_mean = tf.matmul(self.Mk, GX_user_mean) - selected
        GX_col_mean = tf.transpose(tf.reduce_mean(tf.multiply(input_mod, x), axis=1, keepdims=True), perm=[0, 2, 1])
        GX_col_mean = tf.matmul(self.Mm, GX_col_mean) - selected
        G_col_tiled = tf.matmul(self.Mm, tf.transpose(input_mod, perm=[0, 2, 1]))
        G_col_tiled = tf.multiply(tf.tile(1.0-tf.eye(self.K), (self.M, 1)), G_col_tiled)
        G_col_mean = tf.reduce_mean(G_col_tiled, axis=2, keepdims=True)
        G_col_max = tf.reduce_max(G_col_tiled, axis=2, keepdims=True)
        G_col_min = tf.reduce_min(G_col_tiled, axis=2, keepdims=True)
        iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)))
        # print(iteration_num.shape)
        num = np.zeros((1, self.k))
        num[0, int(step)] = 1
        iteration_num = tf.tile(iteration_num, (1, 1, self.k)) + num
        # x = tf.reduce_sum(x, axis=2)
        # x = tf.keras.layers.Reshape((self.K*self.M, ))(x)
        # x = tf.reduce_sum(x, axis=1, keepdims=True)
        # x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        row_choice = tf.reduce_sum(x, axis=2, keepdims=True)
        row_choice = tf.matmul(self.Mk, row_choice)
        row_choice = row_choice - tf.keras.layers.Reshape((self.M*self.K, 1))(x)
        col_choice = tf.transpose(tf.reduce_sum(x, axis=1, keepdims=True), perm=[0,2,1])
        col_choice = tf.matmul(self.Mm, col_choice)
        col_choice = col_choice - tf.keras.layers.Reshape((self.M*self.K, 1))(x)
        # iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)) + tf.constant(step))
        input_i = input_concatnator(
            [input_reshaper(input_mod), selected,
             G_min, G_mean,
             G_user_mean, G_user_max, G_user_min,
             G_col_max, G_col_min, G_col_mean,
             -interference_t, -interference_f, -interference_f_2,
             -GX_user_mean, -GX_col_mean,
             -row_choice,
             -col_choice,
             iteration_num])
        # print(input_i[:, 0, 4])
        # print(input_i[:, 0, 5])
        # print(input_i[:, 0, 6])
        # print(input_i[:, 0, 10])
        # print(input_i[:, 0, 11])
        # print(input_i[:, 0, 12])

        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Sequential_Per_link_Input_modification_most_G_raw_self",
            'Mk': None,
            'Mm': None
        })
        return config
class Per_link_Input_modification_most_G_raw_self(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Per_link_Input_modification_most_G_raw_self, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, x_raw, input_mod, step):
        if self.Mk is None:
            self.Mk = np.zeros((self.K*self.M, self.K), dtype=np.float32)
            self.Mm = np.zeros((self.K*self.M, self.M), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.M):
                    self.Mk[i*self.M+j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.K):
                    self.Mm[i*self.K+j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)
        x = tf.reduce_sum(tf.keras.layers.Softmax(axis=1)(x_raw), axis=2)
        x = tf.keras.layers.Reshape((self.K, self.M))(x)
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
        power = tf.tile(tf.expand_dims(tf.reduce_sum(input_mod, axis=1), 1), (1, self.K, 1)) - input_mod
        interference_f = tf.multiply(power, x)
        up = tf.multiply(input_mod, x)
        interference_f_2 = tf.tile(tf.reduce_sum(up, axis=1, keepdims=True), (1, self.K, 1)) - up
        selected = tf.keras.layers.Reshape((self.M*self.K, 1))(tf.multiply(x, input_mod))
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        interference_t = tf.matmul(input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
        interference_t = input_reshaper(interference_t)
        interference_f = input_reshaper(interference_f)
        interference_f_2 = input_reshaper(interference_f_2)
        G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M*self.K, ))(input_mod), axis=1, keepdims=True)
        G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, self.K * self.M, 1))
        G_user_tiled = tf.matmul(self.Mk, input_mod)
        G_user_tiled = tf.multiply(tf.tile(1.0-tf.eye(self.M), (self.K, 1)), G_user_tiled)
        G_user_mean = tf.reduce_mean(G_user_tiled, axis=2, keepdims=True)
        G_user_max = tf.reduce_max(G_user_tiled, axis=2, keepdims=True)
        G_user_min = tf.reduce_min(G_user_tiled, axis=2, keepdims=True)
        GX_user_mean = tf.reduce_mean(tf.multiply(input_mod, x), axis=2, keepdims=True)
        GX_user_mean = tf.matmul(self.Mk, GX_user_mean) - selected
        GX_col_mean = tf.transpose(tf.reduce_mean(tf.multiply(input_mod, x), axis=1, keepdims=True), perm=[0, 2, 1])
        GX_col_mean = tf.matmul(self.Mm, GX_col_mean) - selected
        G_col_tiled = tf.matmul(self.Mm, tf.transpose(input_mod, perm=[0, 2, 1]))
        G_col_tiled = tf.multiply(tf.tile(1.0-tf.eye(self.K), (self.M, 1)), G_col_tiled)
        G_col_mean = tf.reduce_mean(G_col_tiled, axis=2, keepdims=True)
        G_col_min = tf.reduce_min(G_col_tiled, axis=2, keepdims=True)
        G_col_max = tf.reduce_max(G_col_tiled, axis=2, keepdims=True)
        # iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)))
        # print(iteration_num.shape)
        # x = tf.reduce_sum(x, axis=2)
        # x = tf.keras.layers.Reshape((self.K*self.M, ))(x)
        # x = tf.reduce_sum(x, axis=1, keepdims=True)
        # x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        row_choice = tf.reduce_sum(x, axis=2, keepdims=True)
        row_choice = tf.matmul(self.Mk, row_choice)
        row_choice = row_choice - tf.keras.layers.Reshape((self.M*self.K, 1))(x)
        col_choice = tf.transpose(tf.reduce_sum(x, axis=1, keepdims=True), perm=[0,2,1])
        col_choice = tf.matmul(self.Mm, col_choice)
        col_choice = col_choice - tf.keras.layers.Reshape((self.M*self.K, 1))(x)
        # iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)) + tf.constant(step))
        input_i = input_concatnator(
            [input_reshaper(input_mod), selected,
             G_mean,
             G_user_mean, G_user_max, G_user_min,
             # G_user_max, G_user_min,
             G_col_mean, G_col_max, G_col_min,
             interference_t, interference_f, interference_f_2,
             GX_user_mean, GX_col_mean,
             col_choice, row_choice,
             x_raw])
        # print(input_i[:, 0, 4])
        # print(input_i[:, 0, 5])
        # print(input_i[:, 0, 6])
        # print(input_i[:, 0, 10])
        # print(input_i[:, 0, 11])
        # print(input_i[:, 0, 12])

        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Per_link_Input_modification_most_G_raw_self",
            'Mk': None,
            'Mm': None
        })
        return config
class Per_link_Input_modification_most_G_raw_self_more_interference(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Per_link_Input_modification_most_G_raw_self_more_interference, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, x_raw, input_mod, step):
        if self.Mk is None:
            self.Mk = np.zeros((self.K*self.M, self.K), dtype=np.float32)
            self.Mm = np.zeros((self.K*self.M, self.M), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.M):
                    self.Mk[i*self.M+j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.K):
                    self.Mm[i*self.K+j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)
        x = tf.reduce_sum(tf.keras.layers.Softmax(axis=1)(x_raw), axis=2)
        x = tf.keras.layers.Reshape((self.K, self.M))(x)
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
        power = tf.tile(tf.expand_dims(tf.reduce_sum(input_mod, axis=1), 1), (1, self.K, 1)) - input_mod
        interference_f = tf.multiply(power, x)
        up = tf.multiply(input_mod, x)
        interference_t_2 = tf.tile(tf.reduce_sum(up, axis=1, keepdims=True), [1,self.K,1])
        interference_t_2 = tf.divide(up, interference_t_2 - up + 1)
        interference_t_2 = input_reshaper(interference_t_2)
        interference_f_2 = tf.tile(tf.reduce_sum(up, axis=1, keepdims=True), (1, self.K, 1)) - up
        selected = tf.keras.layers.Reshape((self.M*self.K, 1))(tf.multiply(x, input_mod))
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        interference_t = tf.matmul(input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
        interference_t = input_reshaper(interference_t)
        interference_f = input_reshaper(interference_f)
        interference_f_2 = input_reshaper(interference_f_2)
        G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M*self.K, ))(input_mod), axis=1, keepdims=True)
        G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, self.K * self.M, 1))
        G_user_tiled = tf.matmul(self.Mk, input_mod)
        G_user_tiled = tf.multiply(tf.tile(1.0-tf.eye(self.M), (self.K, 1)), G_user_tiled)
        G_user_mean = tf.reduce_mean(G_user_tiled, axis=2, keepdims=True)
        G_user_max = tf.reduce_max(G_user_tiled, axis=2, keepdims=True)
        G_user_min = tf.reduce_min(G_user_tiled, axis=2, keepdims=True)
        GX_user_mean = tf.reduce_mean(tf.multiply(input_mod, x), axis=2, keepdims=True)
        GX_user_mean = tf.matmul(self.Mk, GX_user_mean) - selected
        GX_col_mean = tf.transpose(tf.reduce_mean(tf.multiply(input_mod, x), axis=1, keepdims=True), perm=[0, 2, 1])
        GX_col_mean = tf.matmul(self.Mm, GX_col_mean) - selected
        G_col_tiled = tf.matmul(self.Mm, tf.transpose(input_mod, perm=[0, 2, 1]))
        G_col_tiled = tf.multiply(tf.tile(1.0-tf.eye(self.K), (self.M, 1)), G_col_tiled)
        G_col_mean = tf.reduce_mean(G_col_tiled, axis=2, keepdims=True)
        G_col_min = tf.reduce_min(G_col_tiled, axis=2, keepdims=True)
        G_col_max = tf.reduce_max(G_col_tiled, axis=2, keepdims=True)
        # iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)))
        # print(iteration_num.shape)
        # x = tf.reduce_sum(x, axis=2)
        # x = tf.keras.layers.Reshape((self.K*self.M, ))(x)
        # x = tf.reduce_sum(x, axis=1, keepdims=True)
        # x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        row_choice = tf.reduce_sum(x, axis=2, keepdims=True)
        row_choice = tf.matmul(self.Mk, row_choice)
        row_choice = row_choice - tf.keras.layers.Reshape((self.M*self.K, 1))(x)
        col_choice = tf.transpose(tf.reduce_sum(x, axis=1, keepdims=True), perm=[0,2,1])
        col_choice = tf.matmul(self.Mm, col_choice)
        col_choice = col_choice - tf.keras.layers.Reshape((self.M*self.K, 1))(x)
        # iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)) + tf.constant(step))
        input_i = input_concatnator(
            [input_reshaper(input_mod), selected,
             G_mean,
             G_user_mean, G_user_max, G_user_min,
             # G_user_max, G_user_min,
             G_col_mean, G_col_max, G_col_min,
             interference_t, interference_f, interference_f_2, interference_t_2,
             GX_user_mean, GX_col_mean,
             col_choice, row_choice,
             x_raw])
        # print(input_i[:, 0, 4])
        # print(input_i[:, 0, 5])
        # print(input_i[:, 0, 6])
        # print(input_i[:, 0, 10])
        # print(input_i[:, 0, 11])
        # print(input_i[:, 0, 12])

        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Per_link_Input_modification_most_G_raw_self_more_interference",
            'Mk': None,
            'Mm': None
        })
        return config
class Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, x_raw, input_mod, step):
        if self.Mk is None:
            self.Mk = np.zeros((self.K*self.M, self.K), dtype=np.float32)
            self.Mm = np.zeros((self.K*self.M, self.M), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.M):
                    self.Mk[i*self.M+j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.K):
                    self.Mm[i*self.K+j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)
        x = tf.reduce_sum(tf.keras.layers.Softmax(axis=1)(x_raw), axis=2)
        x = tf.keras.layers.Reshape((self.K, self.M))(x)
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
        power = tf.tile(tf.expand_dims(tf.reduce_sum(input_mod, axis=1), 1), (1, self.K, 1)) - input_mod
        interference_f = tf.multiply(power, x)
        up = tf.multiply(input_mod, x)
        interference_t_2 = tf.tile(tf.reduce_sum(up, axis=1, keepdims=True), [1,self.K,1])
        interference_t_2 = tf.divide(up, interference_t_2 - up + 1)
        interference_t_2 = input_reshaper(interference_t_2)
        interference_f_2 = tf.tile(tf.reduce_sum(up, axis=1, keepdims=True), (1, self.K, 1)) - up
        selected = tf.keras.layers.Reshape((self.M*self.K, 1))(tf.multiply(x, input_mod))
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        interference_t = tf.matmul(input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
        interference_t = input_reshaper(interference_t)
        interference_f = input_reshaper(interference_f)
        interference_f_2 = input_reshaper(interference_f_2)
        G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M*self.K, ))(input_mod), axis=1, keepdims=True)
        G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, self.K * self.M, 1))
        G_user_tiled = tf.matmul(self.Mk, input_mod)
        G_user_tiled = tf.multiply(tf.tile(1.0-tf.eye(self.M), (self.K, 1)), G_user_tiled)
        G_user_mean = tf.reduce_mean(G_user_tiled, axis=2, keepdims=True)
        G_user_max = tf.reduce_max(G_user_tiled, axis=2, keepdims=True)
        G_user_min = tf.reduce_min(G_user_tiled, axis=2, keepdims=True)
        GX_user_mean = tf.reduce_sum(tf.multiply(input_mod, x), axis=2, keepdims=True)
        GX_user_mean = tf.matmul(self.Mk, GX_user_mean) - selected
        GX_col_mean = tf.transpose(tf.reduce_sum(tf.multiply(input_mod, x), axis=1, keepdims=True), perm=[0, 2, 1])
        GX_col_mean = tf.matmul(self.Mm, GX_col_mean) - selected
        G_col_tiled = tf.matmul(self.Mm, tf.transpose(input_mod, perm=[0, 2, 1]))
        G_col_tiled = tf.multiply(tf.tile(1.0-tf.eye(self.K), (self.M, 1)), G_col_tiled)
        G_col_mean = tf.reduce_mean(G_col_tiled, axis=2, keepdims=True)
        G_col_min = tf.reduce_min(G_col_tiled, axis=2, keepdims=True)
        G_col_max = tf.reduce_max(G_col_tiled, axis=2, keepdims=True)
        # iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)))
        # print(iteration_num.shape)
        # x = tf.reduce_sum(x, axis=2)
        # x = tf.keras.layers.Reshape((self.K*self.M, ))(x)
        # x = tf.reduce_sum(x, axis=1, keepdims=True)
        # x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        row_choice = tf.reduce_sum(x, axis=2, keepdims=True)
        row_choice = tf.matmul(self.Mk, row_choice)
        row_choice = row_choice - tf.keras.layers.Reshape((self.M*self.K, 1))(x)
        col_choice = tf.transpose(tf.reduce_sum(x, axis=1, keepdims=True), perm=[0,2,1])
        col_choice = tf.matmul(self.Mm, col_choice)
        col_choice = col_choice - tf.keras.layers.Reshape((self.M*self.K, 1))(x)
        # iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)) + tf.constant(step))
        input_i = input_concatnator(
            [input_reshaper(input_mod), selected,
             G_mean,
             G_user_mean, G_user_max, G_user_min,
             # G_user_max, G_user_min,
             G_col_mean, G_col_max, G_col_min,
             interference_t, interference_f, interference_f_2, interference_t_2,
             GX_user_mean, GX_col_mean,
             col_choice, row_choice,
             x_raw])
        # print(input_i[:, 0, 4])
        # print(input_i[:, 0, 5])
        # print(input_i[:, 0, 6])
        # print(input_i[:, 0, 10])
        # print(input_i[:, 0, 11])
        # print(input_i[:, 0, 12])

        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum",
            'Mk': None,
            'Mm': None
        })
        return config
class Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, x_raw, input_mod, weights):
        if self.Mk is None:
            self.Mk = np.zeros((self.K*self.M, self.K), dtype=np.float32)
            self.Mm = np.zeros((self.K*self.M, self.M), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.M):
                    self.Mk[i*self.M+j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.K):
                    self.Mm[i*self.K+j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)
        x = tf.reduce_sum(tf.keras.layers.Softmax(axis=1)(x_raw), axis=2)
        x = tf.keras.layers.Reshape((self.K, self.M))(x)
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
        power = tf.tile(tf.expand_dims(tf.reduce_sum(input_mod, axis=1), 1), (1, self.K, 1)) - input_mod # interference from each node
        interference_f = tf.multiply(power, x)
        # add weight related modification
        interference_f = tf.multiply(interference_f, tf.tile(weights, [1, 1, self.M]))
        up = tf.multiply(input_mod, x)
        # add weight related modification
        up = tf.multiply(up, tf.tile(weights, [1, 1, self.M]))
        interference_t_2 = tf.tile(tf.reduce_sum(up, axis=1, keepdims=True), [1,self.K,1])
        interference_t_2 = tf.divide(up, interference_t_2 - up + 1)
        interference_t_2 = input_reshaper(interference_t_2)
        interference_f_2 = tf.tile(tf.reduce_sum(up, axis=1, keepdims=True), (1, self.K, 1)) - up
        selected = tf.keras.layers.Reshape((self.M*self.K, 1))(tf.multiply(x, input_mod))
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        weighted_input_mod = tf.multiply(input_mod, tf.tile(weights, [1, 1, self.M]))


        interference_t = tf.matmul(weighted_input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
        interference_t = input_reshaper(interference_t)
        interference_f = input_reshaper(interference_f)
        interference_f_2 = input_reshaper(interference_f_2)

        G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M*self.K, ))(weighted_input_mod), axis=1, keepdims=True)
        G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, self.K * self.M, 1))
        G_user_tiled = tf.matmul(self.Mk, weighted_input_mod)
        G_user_tiled = tf.multiply(tf.tile(1.0-tf.eye(self.M), (self.K, 1)), G_user_tiled)
        G_user_mean = tf.reduce_mean(G_user_tiled, axis=2, keepdims=True)
        G_user_max = tf.reduce_max(G_user_tiled, axis=2, keepdims=True)
        G_user_min = tf.reduce_min(G_user_tiled, axis=2, keepdims=True)
        GX_user_mean = tf.reduce_sum(tf.multiply(weighted_input_mod, x), axis=2, keepdims=True)
        GX_user_mean = tf.matmul(self.Mk, GX_user_mean) - selected
        GX_col_mean = tf.transpose(tf.reduce_sum(tf.multiply(weighted_input_mod, x), axis=1, keepdims=True), perm=[0, 2, 1])
        GX_col_mean = tf.matmul(self.Mm, GX_col_mean) - selected
        G_col_tiled = tf.matmul(self.Mm, tf.transpose(weighted_input_mod, perm=[0, 2, 1]))
        G_col_tiled = tf.multiply(tf.tile(1.0-tf.eye(self.K), (self.M, 1)), G_col_tiled)
        G_col_mean = tf.reduce_mean(G_col_tiled, axis=2, keepdims=True)
        G_col_min = tf.reduce_min(G_col_tiled, axis=2, keepdims=True)
        G_col_max = tf.reduce_max(G_col_tiled, axis=2, keepdims=True)
        # iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)))
        # print(iteration_num.shape)
        # x = tf.reduce_sum(x, axis=2)
        # x = tf.keras.layers.Reshape((self.K*self.M, ))(x)
        # x = tf.reduce_sum(x, axis=1, keepdims=True)
        # x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        row_choice = tf.reduce_sum(x, axis=2, keepdims=True)
        row_choice = tf.matmul(self.Mk, row_choice)
        row_choice = row_choice - tf.keras.layers.Reshape((self.M*self.K, 1))(x)
        col_choice = tf.transpose(tf.reduce_sum(x, axis=1, keepdims=True), perm=[0,2,1])
        col_choice = tf.matmul(self.Mm, col_choice)
        col_choice = col_choice - tf.keras.layers.Reshape((self.M*self.K, 1))(x)
        # iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)) + tf.constant(step))
        weights = tf.matmul(self.Mk, weights)

        input_i = input_concatnator(
            [input_reshaper(input_mod), selected, weights,
             G_mean,
             G_user_mean, G_user_max, G_user_min,
             # G_user_max, G_user_min,
             G_col_mean, G_col_max, G_col_min,
             interference_t, interference_f, interference_f_2, interference_t_2,
             GX_user_mean, GX_col_mean,
             col_choice, row_choice,
             x_raw])
        # print(input_i[:, 0, 4])
        # print(input_i[:, 0, 5])
        # print(input_i[:, 0, 6])
        # print(input_i[:, 0, 10])
        # print(input_i[:, 0, 11])
        # print(input_i[:, 0, 12])

        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights",
            'Mk': None,
            'Mm': None
        })
        return config
class Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights_different_weights(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights_different_weights, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, x_raw, input_mod, weights):
        if self.Mk is None:
            self.Mk = np.zeros((self.K*self.M, self.K), dtype=np.float32)
            self.Mm = np.zeros((self.K*self.M, self.M), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.M):
                    self.Mk[i*self.M+j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.K):
                    self.Mm[i*self.K+j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)
        x = tf.reduce_sum(tf.keras.layers.Softmax(axis=1)(x_raw), axis=2)
        x = tf.keras.layers.Reshape((self.K, self.M))(x)
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
        power = tf.tile(tf.expand_dims(tf.reduce_sum(input_mod, axis=1), 1), (1, self.K, 1)) - input_mod # interference from each node
        interference_f = tf.multiply(power, x)
        # add weight related modification
        interference_f = tf.multiply(interference_f, tf.tile(weights, [1, 1, self.M]))
        up = tf.multiply(input_mod, x)
        # add weight related modification
        up = tf.multiply(up, tf.tile(weights, [1, 1, self.M]))
        interference_t_2 = tf.tile(tf.reduce_sum(up, axis=1, keepdims=True), [1,self.K,1])
        interference_t_2 = tf.divide(up, interference_t_2 - up + 1)
        interference_t_2 = input_reshaper(interference_t_2)
        interference_f_2 = tf.tile(tf.reduce_sum(up, axis=1, keepdims=True), (1, self.K, 1)) - up
        selected = tf.keras.layers.Reshape((self.M*self.K, 1))(tf.multiply(x, input_mod))
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        weighted_input_mod = tf.multiply(input_mod, tf.tile(weights, [1, 1, self.M]))


        interference_t = tf.matmul(weighted_input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
        interference_t = input_reshaper(interference_t)
        interference_f = input_reshaper(interference_f)
        interference_f_2 = input_reshaper(interference_f_2)

        G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M*self.K, ))(weighted_input_mod), axis=1, keepdims=True)
        G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, self.K * self.M, 1))
        G_user_tiled = tf.matmul(self.Mk, weighted_input_mod)
        G_user_tiled = tf.multiply(tf.tile(1.0-tf.eye(self.M), (self.K, 1)), G_user_tiled)
        G_user_mean = tf.reduce_mean(G_user_tiled, axis=2, keepdims=True)
        G_user_max = tf.reduce_max(G_user_tiled, axis=2, keepdims=True)
        G_user_min = tf.reduce_min(G_user_tiled, axis=2, keepdims=True)
        GX_user_mean = tf.reduce_sum(tf.multiply(weighted_input_mod, x), axis=2, keepdims=True)
        GX_user_mean = tf.matmul(self.Mk, GX_user_mean) - selected
        GX_col_mean = tf.transpose(tf.reduce_sum(tf.multiply(weighted_input_mod, x), axis=1, keepdims=True), perm=[0, 2, 1])
        GX_col_mean = tf.matmul(self.Mm, GX_col_mean) - selected
        G_col_tiled = tf.matmul(self.Mm, tf.transpose(weighted_input_mod, perm=[0, 2, 1]))
        G_col_tiled = tf.multiply(tf.tile(1.0-tf.eye(self.K), (self.M, 1)), G_col_tiled)
        G_col_mean = tf.reduce_mean(G_col_tiled, axis=2, keepdims=True)
        G_col_min = tf.reduce_min(G_col_tiled, axis=2, keepdims=True)
        G_col_max = tf.reduce_max(G_col_tiled, axis=2, keepdims=True)
        # iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)))
        # print(iteration_num.shape)
        # x = tf.reduce_sum(x, axis=2)
        # x = tf.keras.layers.Reshape((self.K*self.M, ))(x)
        # x = tf.reduce_sum(x, axis=1, keepdims=True)
        # x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        row_choice = tf.reduce_sum(x, axis=2, keepdims=True)
        row_choice = tf.matmul(self.Mk, row_choice)
        row_choice = row_choice - tf.keras.layers.Reshape((self.M*self.K, 1))(x)
        col_choice = tf.transpose(tf.reduce_sum(x, axis=1, keepdims=True), perm=[0,2,1])
        col_choice = tf.matmul(self.Mm, col_choice)
        col_choice = col_choice - tf.keras.layers.Reshape((self.M*self.K, 1))(x)
        # iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)) + tf.constant(step))
        # assuming weights are of shape (N, K, 1)
        tiled_weights = tf.tile(tf.expand_dims(weights, axis=1), [1, self.K, 1, 1])
        onesK = tf.expand_dims(tf.expand_dims(tf.eye(self.K), axis=0), axis=3) # this has the shape of (1, K, K, 1)
        tiled_weights = tiled_weights - tf.multiply(onesK, tiled_weights) # reduce the own link from it
        weights_max = tf.reduce_max(tiled_weights, axis=2) # (N, K, 1)
        weights_min = tf.reduce_min(tiled_weights, axis=2)
        weights_mean = tf.reduce_mean(tiled_weights, axis=2)
        self_weights = tf.matmul(self.Mk, weights)
        weights_max = tf.matmul(self.Mk, weights_max)
        weights_min = tf.matmul(self.Mk, weights_min)
        weights_mean = tf.matmul(self.Mk, weights_mean)

        input_i = input_concatnator(
            [input_reshaper(input_mod), selected, self_weights,
             weights_max, weights_min, weights_mean,
             G_mean,
             G_user_mean, G_user_max, G_user_min,
             # G_user_max, G_user_min,
             G_col_mean, G_col_max, G_col_min,
             interference_t, interference_f, interference_f_2, interference_t_2,
             GX_user_mean, GX_col_mean,
             col_choice, row_choice,
             x_raw])
        # print(input_i[:, 0, 4])
        # print(input_i[:, 0, 5])
        # print(input_i[:, 0, 6])
        # print(input_i[:, 0, 10])
        # print(input_i[:, 0, 11])
        # print(input_i[:, 0, 12])

        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights_different_weights",
            'Mk': None,
            'Mm': None
        })
        return config

class Per_link_Input_modification_most_G_raw_self_more_choice(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Per_link_Input_modification_most_G_raw_self_more_choice, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, x_raw, input_mod, step):
        if self.Mk is None:
            self.Mk = np.zeros((self.K*self.M, self.K), dtype=np.float32)
            self.Mm = np.zeros((self.K*self.M, self.M), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.M):
                    self.Mk[i*self.M+j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.K):
                    self.Mm[i*self.K+j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)
        x = tf.reduce_sum(tf.keras.layers.Softmax(axis=1)(x_raw), axis=2)
        x = tf.keras.layers.Reshape((self.K, self.M))(x)
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
        power = tf.tile(tf.expand_dims(tf.reduce_sum(input_mod, axis=1), 1), (1, self.K, 1)) - input_mod
        interference_f = tf.multiply(power, x)
        up = tf.multiply(input_mod, x)
        interference_f_2 = tf.tile(tf.reduce_sum(up, axis=1, keepdims=True), (1, self.K, 1)) - up
        selected = tf.keras.layers.Reshape((self.M*self.K, 1))(tf.multiply(x, input_mod))
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        interference_t = tf.matmul(input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
        interference_t = input_reshaper(interference_t)
        interference_f = input_reshaper(interference_f)
        interference_f_2 = input_reshaper(interference_f_2)
        G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M*self.K, ))(input_mod), axis=1, keepdims=True)
        G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, self.K * self.M, 1))
        G_user_tiled = tf.matmul(self.Mk, input_mod)
        G_user_tiled = tf.multiply(tf.tile(1.0-tf.eye(self.M), (self.K, 1)), G_user_tiled)
        G_user_mean = tf.reduce_mean(G_user_tiled, axis=2, keepdims=True)
        G_user_max = tf.reduce_max(G_user_tiled, axis=2, keepdims=True)
        G_user_min = tf.reduce_min(G_user_tiled, axis=2, keepdims=True)
        GX_user_mean = tf.reduce_mean(tf.multiply(input_mod, x), axis=2, keepdims=True)
        GX_user_mean = tf.matmul(self.Mk, GX_user_mean) - selected
        GX_col_mean = tf.transpose(tf.reduce_mean(tf.multiply(input_mod, x), axis=1, keepdims=True), perm=[0, 2, 1])
        GX_col_mean = tf.matmul(self.Mm, GX_col_mean) - selected
        G_col_tiled = tf.matmul(self.Mm, tf.transpose(input_mod, perm=[0, 2, 1]))
        G_col_tiled = tf.multiply(tf.tile(1.0-tf.eye(self.K), (self.M, 1)), G_col_tiled)
        G_col_mean = tf.reduce_mean(G_col_tiled, axis=2, keepdims=True)
        G_col_min = tf.reduce_min(G_col_tiled, axis=2, keepdims=True)
        G_col_max = tf.reduce_max(G_col_tiled, axis=2, keepdims=True)
        # iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)))
        # print(iteration_num.shape)
        # x = tf.reduce_sum(x, axis=2)
        # x = tf.keras.layers.Reshape((self.K*self.M, ))(x)
        # x = tf.reduce_sum(x, axis=1, keepdims=True)
        # x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        row_choice = tf.reduce_sum(x, axis=2, keepdims=True)
        all_choice = tf.reduce_sum(x, axis=1)[:, 0]
        row_choice = tf.matmul(self.Mk, row_choice)
        row_choice = row_choice - tf.keras.layers.Reshape((self.M*self.K, 1))(x)
        col_choice = tf.transpose(tf.reduce_sum(x, axis=1, keepdims=True), perm=[0,2,1])
        col_choice = tf.matmul(self.Mm, col_choice)
        col_choice = col_choice - tf.keras.layers.Reshape((self.M*self.K, 1))(x)
        # iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)) + tf.constant(step))
        all_choice = all_choice - x
        input_i = input_concatnator(
            [input_reshaper(input_mod), selected,
             G_mean,
             G_user_mean, G_user_max, G_user_min,
             # G_user_max, G_user_min,
             G_col_mean, G_col_max, G_col_min,
             interference_t, interference_f, interference_f_2,
             GX_user_mean, GX_col_mean,
             col_choice, row_choice,
             x_raw])
        # print(input_i[:, 0, 4])
        # print(input_i[:, 0, 5])
        # print(input_i[:, 0, 6])
        # print(input_i[:, 0, 10])
        # print(input_i[:, 0, 11])
        # print(input_i[:, 0, 12])

        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Per_link_Input_modification_most_G_raw_self_more_choice",
            'Mk': None,
            'Mm': None
        })
        return config
class Per_link_Input_modification_most_G_original(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Per_link_Input_modification_most_G_original, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, x_raw, input_mod, step):
        if self.Mk is None:
            self.Mk = np.zeros((self.K*self.M, self.K), dtype=np.float32)
            self.Mm = np.zeros((self.K*self.M, self.M), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.M):
                    self.Mk[i*self.M+j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.K):
                    self.Mm[i*self.K+j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)
        x = tf.reduce_sum(tf.keras.layers.Softmax(axis=1)(x_raw), axis=2)
        x = tf.keras.layers.Reshape((self.K, self.M))(x)
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
        power = tf.tile(tf.expand_dims(tf.reduce_sum(input_mod, axis=1), 1), (1, self.K, 1)) - input_mod
        interference_f = tf.multiply(power, x)
        up = tf.multiply(input_mod, x)
        interference_f_2 = tf.tile(tf.reduce_sum(up, axis=1, keepdims=True), (1, self.K, 1)) - up
        selected = tf.keras.layers.Reshape((self.M*self.K, 1))(tf.multiply(x, input_mod))
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        interference_t = tf.matmul(input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
        interference_t = input_reshaper(interference_t)
        interference_f = input_reshaper(interference_f)
        interference_f_2 = input_reshaper(interference_f_2)
        # iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)) + tf.constant(step))
        x = tf.keras.layers.Reshape((self.K * self.M,))(x)
        # x = tf.reduce_sum(x, axis=1, keepdims=True)
        x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        input_i = input_concatnator(
            [input_reshaper(input_mod), selected,
             # G_user_max, G_user_min,
             interference_t, interference_f, interference_f_2,
             x])
        # print(input_i[:, 0, 4])
        # print(input_i[:, 0, 5])
        # print(input_i[:, 0, 6])
        # print(input_i[:, 0, 10])
        # print(input_i[:, 0, 11])
        # print(input_i[:, 0, 12])

        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Per_link_Input_modification_most_G_original",
            'Mk': None,
            'Mm': None
        })
        return config
class Per_link_Input_modification_most_G_raw_self_sigmoid(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Per_link_Input_modification_most_G_raw_self_sigmoid, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, x_raw, input_mod, step):
        if self.Mk is None:
            self.Mk = np.zeros((self.K*self.M, self.K), dtype=np.float32)
            self.Mm = np.zeros((self.K*self.M, self.M), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.M):
                    self.Mk[i*self.M+j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.K):
                    self.Mm[i*self.K+j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)
        x = tf.reduce_sum(x_raw, axis=2)
        x = tf.keras.layers.Reshape((self.K, self.M))(x)
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
        power = tf.tile(tf.expand_dims(tf.reduce_sum(input_mod, axis=1), 1), (1, self.K, 1)) - input_mod
        interference_f = tf.multiply(power, x)
        up = tf.multiply(input_mod, x)
        interference_f_2 = tf.tile(tf.reduce_sum(up, axis=1, keepdims=True), (1, self.K, 1)) - up
        selected = tf.keras.layers.Reshape((self.M*self.K, 1))(tf.multiply(x, input_mod))
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        interference_t = tf.matmul(input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
        interference_t = input_reshaper(interference_t)
        interference_f = input_reshaper(interference_f)
        interference_f_2 = input_reshaper(interference_f_2)
        G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M*self.K, ))(input_mod), axis=1, keepdims=True)
        G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, self.K * self.M, 1))
        G_user_tiled = tf.matmul(self.Mk, input_mod)
        G_user_tiled = tf.multiply(tf.tile(1.0-tf.eye(self.M), (self.K, 1)), G_user_tiled)
        G_user_mean = tf.reduce_mean(G_user_tiled, axis=2, keepdims=True)
        G_user_max = tf.reduce_max(G_user_tiled, axis=2, keepdims=True)
        G_user_min = tf.reduce_min(G_user_tiled, axis=2, keepdims=True)
        GX_user_mean = tf.reduce_mean(tf.multiply(input_mod, x), axis=2, keepdims=True)
        GX_user_mean = tf.matmul(self.Mk, GX_user_mean) - selected
        GX_col_mean = tf.transpose(tf.reduce_mean(tf.multiply(input_mod, x), axis=1, keepdims=True), perm=[0, 2, 1])
        GX_col_mean = tf.matmul(self.Mm, GX_col_mean) - selected
        G_col_tiled = tf.matmul(self.Mm, tf.transpose(input_mod, perm=[0, 2, 1]))
        G_col_tiled = tf.multiply(tf.tile(1.0-tf.eye(self.K), (self.M, 1)), G_col_tiled)
        G_col_mean = tf.reduce_mean(G_col_tiled, axis=2, keepdims=True)
        # G_col_min = tf.reduce_min(G_col_tiled, axis=2, keepdims=True)
        # iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)))
        # print(iteration_num.shape)
        # x = tf.reduce_sum(x, axis=2)
        # x = tf.keras.layers.Reshape((self.K*self.M, ))(x)
        # x = tf.reduce_sum(x, axis=1, keepdims=True)
        # x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        # row_choice = tf.reduce_sum(x, axis=2, keepdims=True)
        # row_choice = tf.matmul(self.Mk, row_choice)
        # row_choice = row_choice - tf.keras.layers.Reshape((self.M*self.K, 1))(x)
        col_choice = tf.transpose(tf.reduce_sum(x, axis=1, keepdims=True), perm=[0,2,1])
        col_choice = tf.matmul(self.Mm, col_choice)
        col_choice = col_choice - tf.keras.layers.Reshape((self.M*self.K, 1))(x)
        # iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)) + tf.constant(step))
        input_i = input_concatnator(
            [input_reshaper(input_mod), selected,
             G_mean,
             G_user_mean, G_user_max, G_user_min,
             # G_user_max, G_user_min,
             G_col_mean,
             interference_t, interference_f, interference_f_2,
             GX_user_mean, GX_col_mean,
             col_choice,
             x_raw])
        # print(input_i[:, 0, 4])
        # print(input_i[:, 0, 5])
        # print(input_i[:, 0, 6])
        # print(input_i[:, 0, 10])
        # print(input_i[:, 0, 11])
        # print(input_i[:, 0, 12])

        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Per_link_Input_modification_most_G_raw_self_sigmoid",
            'Mk': None,
            'Mm': None
        })
        return config
class Per_link_sequential_modification(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Per_link_sequential_modification, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, x, input_mod, step):
        if self.Mk is None:
            self.Mk = np.zeros((self.K*self.M, self.K), dtype=np.float32)
            self.Mm = np.zeros((self.K*self.M, self.M), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.M):
                    self.Mk[i*self.M+j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.K):
                    self.Mm[i*self.K+j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)

        original_x = x # x is shaped [none, K, M]
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
        power = tf.tile(tf.expand_dims(tf.reduce_sum(input_mod, axis=1), 1), (1, self.K, 1)) - input_mod
        interference_f = tf.multiply(power, x)
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        interference_t = tf.matmul(input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
        interference_t = input_reshaper(interference_t)
        interference_f = input_reshaper(interference_f)
        x_user_choice = tf.matmul(self.Mk, tf.reduce_sum(x, axis=2,keepdims=True))[:, :, 0]
        x_user_choice = tf.keras.layers.Reshape((self.K, self.M))(x_user_choice)
        input_mod = input_mod * (1 - x_user_choice)
        G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M*self.K, ))(input_mod), axis=1, keepdims=True)
        G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, self.K * self.M, 1))
        G_max = tf.reduce_max(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
        G_max = tf.tile(tf.expand_dims(G_max, axis=1), (1, self.K * self.M, 1))
        G_min = tf.reduce_min(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
        G_min = tf.tile(tf.expand_dims(G_min, axis=1), (1, self.K * self.M, 1))
        G_user_mean = tf.reduce_mean(input_mod, axis=2, keepdims=True)
        G_user_mean = tf.matmul(self.Mk, G_user_mean)
        G_user_max = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_max = tf.matmul(self.Mk, G_user_max)
        G_user_min = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_min = tf.matmul(self.Mk, G_user_min)
        G_col_mean = tf.transpose(tf.reduce_mean(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        G_col_mean = tf.matmul(self.Mm, G_col_mean)
        G_col_max = tf.transpose(tf.reduce_max(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        G_col_max = tf.matmul(self.Mm, G_col_max)
        G_col_min = tf.transpose(tf.reduce_max(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        G_col_min = tf.matmul(self.Mm, G_col_min)

        x = tf.keras.layers.Reshape((self.K*self.M, ))(x)
        # x = tf.reduce_sum(x, axis=1, keepdims=True)
        x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        #

        # self_decision = tf.keras.layers.Reshape((self.K * self.M, 1))(x)
        # same_user_decision = tf.matmeul(self.Mk, x)
        # x = tf.reduce_sum(x, axis=2)
        # # x = tf.keras.layers.Reshape((self.K*self.M, ))(x)
        # # x = tf.reduce_sum(x, axis=1, keepdims=True)
        # x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        input_i = input_concatnator(
            [input_reshaper(input_mod),
             G_mean, G_max, G_min,
             # G_mean,
             G_user_mean, G_user_min, G_user_max,
             G_col_max, G_col_min, G_col_mean,
             interference_t, interference_f,
             x])
        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Per_link_sequential_modification",
            'Mk': None,
            'Mm': None
        })
        return config
class Per_link_sequential_modification_compressedX(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Per_link_sequential_modification_compressedX, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, x, input_mod, step):
        if self.Mk is None:
            self.Mk = np.zeros((self.K*self.M, self.K), dtype=np.float32)
            self.Mm = np.zeros((self.K*self.M, self.M), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.M):
                    self.Mk[i*self.M+j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.K):
                    self.Mm[i*self.K+j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)

        # original_x = x # x is shaped [none, K, M]
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
        power = tf.tile(tf.expand_dims(tf.reduce_sum(input_mod, axis=1), 1), (1, self.K, 1)) - input_mod
        interference_f = tf.multiply(power, x)
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        interference_t = tf.matmul(input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
        interference_t = input_reshaper(interference_t)
        interference_f = input_reshaper(interference_f)
        x_user_choice = tf.matmul(self.Mk, tf.reduce_sum(x, axis=2,keepdims=True))[:, :, 0]
        x_user_choice = tf.keras.layers.Reshape((self.K, self.M))(x_user_choice)
        G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M*self.K, ))(input_mod), axis=1, keepdims=True)
        G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, self.K * self.M, 1))
        G_max = tf.reduce_max(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
        G_max = tf.tile(tf.expand_dims(G_max, axis=1), (1, self.K * self.M, 1))
        G_min = tf.reduce_min(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
        G_min = tf.tile(tf.expand_dims(G_min, axis=1), (1, self.K * self.M, 1))
        G_user_mean = tf.reduce_mean(input_mod, axis=2, keepdims=True)
        G_user_mean = tf.matmul(self.Mk, G_user_mean)
        G_user_max = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_max = tf.matmul(self.Mk, G_user_max)
        G_user_min = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_min = tf.matmul(self.Mk, G_user_min)
        G_col_mean = tf.transpose(tf.reduce_mean(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        G_col_mean = tf.matmul(self.Mm, G_col_mean)
        G_col_max = tf.transpose(tf.reduce_max(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        G_col_max = tf.matmul(self.Mm, G_col_max)
        G_col_min = tf.transpose(tf.reduce_max(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        G_col_min = tf.matmul(self.Mm, G_col_min)
        # x = tf.keras.layers.Reshape((self.K*self.M, ))(x)
        # # x = tf.reduce_sum(x, axis=1, keepdims=True)
        # x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        input_mod = input_mod * (1.0 - x_user_choice)
        # self_decision = tf.keras.layers.Reshape((self.K * self.M, 1))(x)
        same_user_decision = tf.matmul(self.Mk, x)
        # x = tf.reduce_sum(x, axis=2)
        # # x = tf.keras.layers.Reshape((self.K*self.M, ))(x)
        # # x = tf.reduce_sum(x, axis=1, keepdims=True)
        # x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        input_i = input_concatnator(
            [input_reshaper(input_mod),
             G_mean, G_max, G_min,
             G_user_mean, G_user_min, G_user_max,
             G_col_max, G_col_min, G_col_mean,
             interference_t, interference_f,
             same_user_decision])
        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Per_link_sequential_modification_compressedX",
            'Mk': None,
            'Mm': None
        })
        return config
class Per_link_Input_modification_learnable_G(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Per_link_Input_modification_learnable_G, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        self.row_picker = self.add_weight(name='row_picker',
                                 shape=(self.M, 20),
                                 trainable=True)
        self.row_picker_2 = self.add_weight(name='row_picker_2',
                                 shape=(20, 4),
                                 trainable=True)
        self.bias_row = self.add_weight(name='row_bias',
                                 shape=(20,),
                                 trainable=True)
        self.col_picker = self.add_weight(name='col_picker',
                                 shape=(self.K, 20),
                                 trainable=True)
        self.bias_col = self.add_weight(name='col_bias',
                                 shape=(20,),
                                 trainable=True)
        self.col_picker_2 = self.add_weight(name='col_picker_2',
                                          shape=(20, 4),
                                          trainable=True)
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, x, input_mod, step):
        if self.Mk is None:
            self.Mk = np.zeros((self.K*self.M, self.K), dtype=np.float32)
            self.Mm = np.zeros((self.K*self.M, self.M), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.M):
                    self.Mk[i*self.M+j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.K):
                    self.Mm[i*self.K+j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
        power = tf.tile(tf.expand_dims(tf.reduce_sum(input_mod, axis=1), 1), (1, self.K, 1)) - input_mod
        interference_f = tf.multiply(power, x)
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        interference_t = tf.matmul(input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
        interference_t = input_reshaper(interference_t)
        interference_f = input_reshaper(interference_f)
        G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M*self.K, ))(input_mod), axis=1, keepdims=True)
        G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, self.K * self.M, 1))
        G_max = tf.reduce_max(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
        G_max = tf.tile(tf.expand_dims(G_max, axis=1), (1, self.K * self.M, 1))
        G_min = tf.reduce_min(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
        G_min = tf.tile(tf.expand_dims(G_min, axis=1), (1, self.K * self.M, 1))
        G_user_mean = tf.reduce_mean(input_mod, axis=2, keepdims=True)
        G_user_mean = tf.matmul(self.Mk, G_user_mean)
        G_user_max = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_max = tf.matmul(self.Mk, G_user_max)
        G_user_min = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_min = tf.matmul(self.Mk, G_user_min)
        G_col_mean = tf.transpose(tf.reduce_mean(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        G_col_mean = tf.matmul(self.Mm, G_col_mean)
        G_col_max = tf.transpose(tf.reduce_max(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        G_col_max = tf.matmul(self.Mm, G_col_max)
        G_col_min = tf.transpose(tf.reduce_max(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        G_col_min = tf.matmul(self.Mm, G_col_min)
        G_user_learned_data = tf.matmul(LeakyReLU()(tf.matmul(input_mod, self.row_picker) + self.bias_row), self.row_picker_2)
        G_user_learned_data = tf.matmul(self.Mk, G_user_learned_data)
        G_col_learned_data = tf.matmul(LeakyReLU()(tf.matmul(tf.transpose(input_mod, perm=[0, 2, 1]), self.col_picker) + self.bias_col), self.col_picker_2)
        G_col_learned_data = tf.matmul(self.Mm, G_col_learned_data)
        # x = tf.reduce_sum(x, axis=2)
        x = tf.keras.layers.Reshape((self.K*self.M, ))(x)
        # x = tf.reduce_sum(x, axis=1, keepdims=True)
        x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)) + tf.constant(step))

        input_i = input_concatnator(
            [input_reshaper(input_mod),
             G_mean,G_max,G_min,
             G_user_learned_data, G_user_min, G_user_max, G_user_mean,
             G_col_max, G_col_min, G_col_learned_data, G_col_mean,
             interference_t, interference_f,
             x,
             iteration_num])
        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Per_link_Input_modification_learnable_G",
            'Mk': None,
            'Mm': None
        })
        return config
class Per_link_Input_modification_even_more_G(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Per_link_Input_modification_even_more_G, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)

    def call(self, x, input_mod, step):
        if self.Mk is None:
            self.Mk = np.zeros((self.K * self.M, self.K), dtype=np.float32)
            self.Mm = np.zeros((self.K * self.M, self.M), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.M):
                    self.Mk[i * self.M + j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.K):
                    self.Mm[i * self.K + j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
        power = tf.tile(tf.expand_dims(tf.reduce_sum(input_mod, axis=1), 1), (1, self.K, 1)) - input_mod
        interference_f = tf.multiply(power, x)
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        interference_t = tf.matmul(input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
        interference_t = input_reshaper(interference_t)
        interference_f = input_reshaper(interference_f)
        G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
        G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, self.K * self.M, 1))
        G_user_mean = tf.reduce_mean(input_mod, axis=2, keepdims=True)
        G_user_mean = tf.matmul(self.Mk, G_user_mean)
        G_user_max = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_max = tf.matmul(self.Mk, G_user_max)
        G_user_min = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_min = tf.matmul(self.Mk, G_user_min)
        # x = tf.reduce_sum(x, axis=2)
        x = tf.keras.layers.Reshape((self.K * self.M,))(x)
        # x = tf.reduce_sum(x, axis=1, keepdims=True)
        x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        iteration_num = tf.stop_gradient(
            tf.multiply(tf.constant(0.0), input_reshaper(input_mod)) + tf.constant(step))

        input_i = input_concatnator(
            [input_reshaper(input_mod), G_user_mean, G_mean, G_user_min, G_user_max, interference_t, interference_f,
             x, iteration_num])
        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Per_link_Input_modification_even_more_G",
            'Mk': None,
            'Mm': None
        })
        return config
class Per_link_Input_modification_compress_XG(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Per_link_Input_modification_compress_XG, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, x, input_mod, step):
        if self.Mk is None:
            self.Mk = np.zeros((self.K*self.M, self.K), dtype=np.float32)
            self.Mm = np.zeros((self.K*self.M, self.M), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.M):
                    self.Mk[i*self.M+j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.K):
                    self.Mm[i*self.K+j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
        power = tf.tile(tf.expand_dims(tf.reduce_sum(input_mod, axis=1), 1), (1, self.K, 1)) - input_mod
        interference_f = tf.multiply(power, x)
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        interference_t = tf.matmul(input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
        interference_t = input_reshaper(interference_t)
        interference_f = input_reshaper(interference_f)
        G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M*self.K, ))(input_mod), axis=1, keepdims=True)
        G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, self.K * self.M, 1))
        G_user_mean = tf.reduce_mean(input_mod, axis=2, keepdims=True)
        G_user_mean = tf.matmul(self.Mk, G_user_mean)
        G_user_max = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_max = tf.matmul(self.Mk, G_user_max)
        G_user_min = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_min = tf.matmul(self.Mk, G_user_min)
        self_decision = tf.keras.layers.Reshape((self.K*self.M, 1))(x)
        x = tf.reduce_sum(x, axis=2)
        # x = tf.keras.layers.Reshape((self.K*self.M, ))(x)
        # x = tf.reduce_sum(x, axis=1, keepdims=True)
        x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        user_pos = tf.multiply(0.0*G_mean + 1.0, tf.constant(self.Mk))
        iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)) + tf.constant(step))

        input_i = input_concatnator(
            [input_reshaper(input_mod),
             G_user_mean,
             G_mean,
             G_user_min,
             G_user_max,
             interference_t, interference_f,
             x, self_decision, user_pos,
             iteration_num])
        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Per_link_Input_modification_compress_XG",
            'Mk': None,
            'Mm': None
        })
        return config
class Per_link_Input_modification_compress_XG_alt(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Per_link_Input_modification_compress_XG_alt, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, x, input_mod, step):
        if self.Mk is None:
            self.Mk = np.zeros((self.K*self.M, self.K), dtype=np.float32)
            self.Mm = np.zeros((self.K*self.M, self.M), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.M):
                    self.Mk[i*self.M+j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.K):
                    self.Mm[i*self.K+j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
        power = tf.tile(tf.expand_dims(tf.reduce_sum(input_mod, axis=1), 1), (1, self.K, 1)) - input_mod
        interference_f = tf.multiply(power, x)
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        interference_t = tf.matmul(input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
        interference_t = input_reshaper(interference_t)
        interference_f = input_reshaper(interference_f)
        G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M*self.K, ))(input_mod), axis=1, keepdims=True)
        G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, self.K * self.M, 1))
        G_user_mean = tf.reduce_mean(input_mod, axis=2, keepdims=True)
        G_user_mean = tf.matmul(self.Mk, G_user_mean)
        G_user_max = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_max = tf.matmul(self.Mk, G_user_max)
        G_user_min = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_min = tf.matmul(self.Mk, G_user_min)
        self_decision = tf.keras.layers.Reshape((self.K*self.M, 1))(x)
        same_user_decision = tf.matmul(self.Mk, x)
        x = tf.reduce_sum(x, axis=2)
        # x = tf.keras.layers.Reshape((self.K*self.M, ))(x)
        # x = tf.reduce_sum(x, axis=1, keepdims=True)
        x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        user_pos = tf.multiply(0.0*G_mean + 1.0, tf.constant(self.Mk))
        iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)) + tf.constant(step))

        input_i = input_concatnator(
            [input_reshaper(input_mod),
             G_user_mean,
             G_mean,
             G_user_min,
             G_user_max,
             interference_t, interference_f,
             x, self_decision, user_pos, same_user_decision,
             iteration_num])
        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Per_link_Input_modification_compress_XG_alt",
            'Mk': None,
            'Mm': None
        })
        return config
class Per_link_Input_modification_compress_XG_alt_2(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Per_link_Input_modification_compress_XG_alt_2, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, x, input_mod, step):
        if self.Mk is None:
            self.Mk = np.zeros((self.K*self.M, self.K), dtype=np.float32)
            self.Mm = np.zeros((self.K*self.M, self.M), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.M):
                    self.Mk[i*self.M+j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.K):
                    self.Mm[i*self.K+j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
        power = tf.tile(tf.expand_dims(tf.reduce_sum(input_mod, axis=1), 1), (1, self.K, 1)) - input_mod
        interference_f = tf.multiply(power, x)
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        interference_t = tf.matmul(input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
        interference_t = input_reshaper(interference_t)
        interference_f = input_reshaper(interference_f)
        G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M*self.K, ))(input_mod), axis=1, keepdims=True)
        G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, self.K * self.M, 1))
        G_max = tf.reduce_max(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
        G_max = tf.tile(tf.expand_dims(G_max, axis=1), (1, self.K * self.M, 1))
        G_min = tf.reduce_min(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
        G_min = tf.tile(tf.expand_dims(G_min, axis=1), (1, self.K * self.M, 1))
        G_user_mean = tf.reduce_mean(input_mod, axis=2, keepdims=True)
        G_user_mean = tf.matmul(self.Mk, G_user_mean)
        G_user_max = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_max = tf.matmul(self.Mk, G_user_max)
        G_user_min = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_min = tf.matmul(self.Mk, G_user_min)
        self_decision = tf.keras.layers.Reshape((self.K*self.M, 1))(x)
        same_user_decision = tf.matmul(self.Mk, x)
        x = tf.reduce_sum(x, axis=2)
        # x = tf.keras.layers.Reshape((self.K*self.M, ))(x)
        # x = tf.reduce_sum(x, axis=1, keepdims=True)
        x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        user_pos = tf.multiply(0.0*G_mean + 1.0, tf.constant(self.Mk))
        iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)) + tf.constant(step))

        input_i = input_concatnator(
            [input_reshaper(input_mod),
             G_user_mean, G_user_min, G_user_max,
             G_mean, G_min, G_max,
             interference_t, interference_f,
             x, self_decision, user_pos, same_user_decision,
             iteration_num])
        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Per_link_Input_modification_compress_XG_alt_2",
            'Mk': None,
            'Mm': None
        })
        return config
class Per_link_Input_modification_more_G_alt_2(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Per_link_Input_modification_more_G_alt_2, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, x, input_mod, step):
        if self.Mk is None:
            self.Mk = np.zeros((self.K*self.M, self.K), dtype=np.float32)
            self.Mm = np.zeros((self.K*self.M, self.M), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.M):
                    self.Mk[i*self.M+j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.K):
                    self.Mm[i*self.K+j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
        power = tf.tile(tf.expand_dims(tf.reduce_sum(input_mod, axis=1), 1), (1, self.K, 1)) - input_mod
        interference_f = tf.multiply(power, x)
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        interference_t = tf.matmul(input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
        interference_t = input_reshaper(interference_t)
        interference_f = input_reshaper(interference_f)
        G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M*self.K, ))(input_mod), axis=1, keepdims=True)
        G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, self.K * self.M, 1))
        G_max = tf.reduce_max(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
        G_max = tf.tile(tf.expand_dims(G_max, axis=1), (1, self.K * self.M, 1))
        G_min = tf.reduce_min(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
        G_min = tf.tile(tf.expand_dims(G_min, axis=1), (1, self.K * self.M, 1))
        G_user_mean = tf.reduce_mean(input_mod, axis=2, keepdims=True)
        G_user_mean = tf.matmul(self.Mk, G_user_mean)
        G_user_max = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_max = tf.matmul(self.Mk, G_user_max)
        G_user_min = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_min = tf.matmul(self.Mk, G_user_min)
        # self_decision = tf.keras.layers.Reshape((self.K*self.M, 1))(x)
        # same_user_decision = tf.matmul(self.Mk, x)
        # x = tf.reduce_sum(x, axis=2)
        x = tf.keras.layers.Reshape((self.K*self.M, ))(x)
        # x = tf.reduce_sum(x, axis=1, keepdims=True)
        x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        # user_pos = tf.multiply(0.0*G_mean + 1.0, tf.constant(self.Mk))
        iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)) + tf.constant(step))

        input_i = input_concatnator(
            [input_reshaper(input_mod),
             G_user_mean, G_user_min, G_user_max,
             G_mean, G_min, G_max,
             interference_t, interference_f,
             x,
             iteration_num])
        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Per_link_Input_modification_more_G_alt_2",
            'Mk': None,
            'Mm': None
        })
        return config
class AllInput_input_mod(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, **kwargs):
        super(AllInput_input_mod, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
    def call(self, x_tm1, G_mod):
        x_tm1 = tf.tile(tf.expand_dims(x_tm1, 1), (1, self.N_rf, 1))
        G_mod = tf.tile(tf.expand_dims(G_mod, 1), (1, self.N_rf, 1))
        x_tm1 = tf.concat((x_tm1, G_mod), axis=2)
        onesies = tf.stop_gradient(0*G_mod[:, :, :self.N_rf] + tf.eye(self.N_rf))
        x_tm1 = tf.concat((x_tm1, onesies), axis=2)
        return x_tm1
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'name': "AllInput_input_mod"
        })
        return config
class Distributed_input_mod(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Distributed_input_mod, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, input_mod):
        if self.Mk is None:
            self.Mk = np.zeros((self.K*self.M, self.K), dtype=np.float32)
            self.Mm = np.zeros((self.K*self.M, self.M), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.M):
                    self.Mk[i*self.M+j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.K):
                    self.Mm[i*self.K+j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
        G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M*self.K, ))(input_mod), axis=1, keepdims=True)
        G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, self.K * self.M, 1))
        G_max = tf.reduce_max(tf.keras.layers.Reshape((self.M*self.K, ))(input_mod), axis=1, keepdims=True)
        G_max = tf.tile(tf.expand_dims(G_max, axis=1), (1, self.K * self.M, 1))
        G_min = tf.reduce_min(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
        G_min = tf.tile(tf.expand_dims(G_min, axis=1), (1, self.K * self.M, 1))
        G_user_mean = tf.reduce_mean(input_mod, axis=2, keepdims=True)
        G_user_mean = tf.matmul(self.Mk, G_user_mean)
        G_user_max = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_max = tf.matmul(self.Mk, G_user_max)
        G_user_min = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_min = tf.matmul(self.Mk, G_user_min)
        G_col_mean = tf.transpose(tf.reduce_mean(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        G_col_mean = tf.matmul(self.Mm, G_col_mean)
        G_col_max = tf.transpose(tf.reduce_max(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        G_col_max = tf.matmul(self.Mm, G_col_max)
        G_col_min = tf.transpose(tf.reduce_max(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        G_col_min = tf.matmul(self.Mm, G_col_min)
        input_i = input_concatnator(
            [input_reshaper(input_mod),
             G_mean, G_max, G_min,
             G_user_mean, G_user_min, G_user_max,
             G_col_max, G_col_min, G_col_mean,])
        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Distributed_input_mod",
            'Mk': None,
            'Mm': None
        })
        return config
class Reduced_output_precoder_input_mod(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Reduced_output_precoder_input_mod, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
    def call(self, x, input_mod):
        if self.Mk is None:
            self.Mk = np.zeros((self.N_rf*self.M, self.N_rf), dtype=np.float32)
            self.Mm = np.zeros((self.N_rf*self.M, self.M), dtype=np.float32)
            for i in range(0, self.N_rf):
                for j in range(0, self.M):
                    self.Mk[i*self.M+j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.N_rf):
                    self.Mm[i*self.N_rf+j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)

        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.N_rf, 1))
        G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M*self.N_rf, ))(input_mod), axis=1, keepdims=True)
        G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, self.N_rf * self.M, 1))
        G_max = tf.reduce_max(tf.keras.layers.Reshape((self.M * self.N_rf,))(input_mod), axis=1, keepdims=True)
        G_max = tf.tile(tf.expand_dims(G_max, axis=1), (1, self.N_rf * self.M, 1))
        G_min = tf.reduce_mean(tf.keras.layers.Reshape((self.M * self.N_rf,))(input_mod), axis=1, keepdims=True)
        G_min = tf.tile(tf.expand_dims(G_min, axis=1), (1, self.N_rf * self.M, 1))
        G_user = tf.matmul(self.Mk, input_mod)
        input_i = input_concatnator(
            [input_reshaper(input_mod),
             G_mean, G_max, G_min,
             G_user
             ])
        return input_i
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Reduced_output_precoder_input_mod",
            'Mk': None,
            'Mm': None
        })
        return config
class Reduced_output_input_mod(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Reduced_output_input_mod, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, input_mod, num):
        # alternatively could be input_mod.T
        if self.Mk is None:
            self.Mk = np.zeros((self.K*self.M, self.K), dtype=np.float32)
            self.Mm = np.zeros((self.K*self.M, self.M), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.M):
                    self.Mk[i*self.M+j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.K):
                    self.Mm[i*self.K+j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)
        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M*self.K, ))(input_mod), axis=1, keepdims=True)
        G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, num, 1))
        G_col_mean = tf.reduce_mean(input_mod, axis=1, keepdims=True)
        G_col_mean = tf.tile(G_col_mean, [1, num, 1])
        G_col_max = tf.reduce_max(input_mod, axis=1, keepdims=True)
        G_col_max = tf.tile(G_col_max, [1, num, 1])
        G_col_min = tf.reduce_min(input_mod, axis=1, keepdims=True)
        G_col_min = tf.tile(G_col_min, [1, num, 1])
        input_i = input_concatnator(
            [input_mod,
             G_mean,
             G_col_mean, G_col_min, G_col_max])
        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Reduced_output_input_mod",
            'Mk': None,
            'Mm': None
        })
        return config
class MaskGen(tf.keras.layers.Layer):
    def __init__(self, top_l, **kwargs):
        super(MaskGen, self).__init__()
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, G):
        # alternatively could be input_mod.T
        K = G.shape[1]
        M = G.shape[2]
        top_values, top_indices = tf.math.top_k(G, k=self.top_l)
        G_copy = np.zeros((top_indices.shape[0], K, M))
        for n in range(0, top_indices.shape[0]):
            for i in range(0, K * self.top_l):
                p_i = int(i % self.top_l)
                user_i = int(tf.floor(i / self.top_l))
                G_copy[n, user_i, int(top_indices[n, user_i, p_i])] = top_indices[n, user_i, p_i]
            G_copy = tf.constant(G_copy, dtype=tf.float32)
        return G_copy

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "MaskGen",
            'Mk': None,
            'Mm': None
        })
        return config
class Per_link_Input_modification_most_G_with_mask(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, **kwargs):
        super(Per_link_Input_modification_most_G_with_mask, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, x, input_mod, input_mod_mask, step):
        if self.Mk is None:
            self.Mk = np.zeros((self.K*self.M, self.K), dtype=np.float32)
            self.Mm = np.zeros((self.K*self.M, self.M), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.M):
                    self.Mk[i*self.M+j, i] = 1.0
            for i in range(0, self.M):
                for j in range(0, self.K):
                    self.Mm[i*self.K+j, i] = 1.0
            # self.Mk = tf.Variable(self.Mk, dtype=tf.float32)
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)

        input_concatnator = tf.keras.layers.Concatenate(axis=2)
        input_reshaper = tf.keras.layers.Reshape((self.M * self.K, 1))
        power = tf.tile(tf.expand_dims(tf.reduce_sum(input_mod, axis=1), 1), (1, self.K, 1)) - input_mod
        interference_f = tf.multiply(power, x)
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        interference_t = tf.matmul(input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, 2), (1, 1, self.M))
        interference_t = input_reshaper(interference_t)
        interference_f = input_reshaper(interference_f)
        G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M*self.K, ))(input_mod), axis=1, keepdims=True)
        G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, self.K * self.M, 1))
        G_max = tf.reduce_max(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
        G_max = tf.tile(tf.expand_dims(G_max, axis=1), (1, self.K * self.M, 1))
        G_min = tf.reduce_min(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
        G_min = tf.tile(tf.expand_dims(G_min, axis=1), (1, self.K * self.M, 1))
        G_user_mean = tf.reduce_mean(input_mod, axis=2, keepdims=True)
        G_user_mean = tf.matmul(self.Mk, G_user_mean)
        G_user_max = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_max = tf.matmul(self.Mk, G_user_max)
        G_user_min = tf.reduce_max(input_mod, axis=2, keepdims=True)
        G_user_min = tf.matmul(self.Mk, G_user_min)
        G_col_mean = tf.transpose(tf.reduce_mean(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        G_col_mean = tf.matmul(self.Mm, G_col_mean)
        G_col_max = tf.transpose(tf.reduce_max(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        G_col_max = tf.matmul(self.Mm, G_col_max)
        G_col_min = tf.transpose(tf.reduce_max(input_mod, axis=1, keepdims=True), perm=[0, 2, 1])
        G_col_min = tf.matmul(self.Mm, G_col_min)
        # G_col = tf.matmul(self.Mk, input_mod)
        # x = tf.reduce_sum(x, axis=2)
        x = tf.keras.layers.Reshape((self.K*self.M, ))(x)
        # x = tf.reduce_sum(x, axis=1, keepdims=True)
        x = tf.tile(tf.expand_dims(x, axis=1), (1, self.K * self.M, 1))
        iteration_num = tf.stop_gradient(tf.multiply(tf.constant(0.0), input_reshaper(input_mod)) + tf.constant(step))
        input_i = input_concatnator(
            [input_reshaper(input_mod_mask),
             G_mean, G_max, G_min,
             # G_mean,
             G_user_mean, G_user_min, G_user_max,
             G_col_max, G_col_min, G_col_mean,
             interference_t, interference_f,
             x,
             iteration_num])
        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'name': "Per_link_Input_modification_most_G_with_mask",
            'Mk': None,
            'Mm': None
        })
        return config
class TopPrecoderPerUserInputMod(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, k, top_l, **kwargs):
        super(TopPrecoderPerUserInputMod, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.k = k
        self.Mk = None
        self.Mm = None
        self.top_l = top_l
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, x, smol_x, input_mod, smol_input_mod, step):
        if self.Mk is None:
            self.Mk = np.zeros((self.K*self.top_l, self.K), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.top_l):
                    self.Mk[i*self.top_l+j, i] = 1.0
            self.Mk = tf.Variable(self.Mk, dtype=tf.float32, trainable=False, name="TopPrecoderPerUserInputMod_Mk")
            # self.Mm = tf.Variable(self.Mm, dtype=tf.float32)

        input_concatnator = tf.keras.layers.Concatenate(axis=2, name="TopPrecoderPerUserInputMod_concat1")
        input_reshaper = tf.keras.layers.Reshape((self.top_l * self.K, 1), name="TopPrecoderPerUserInputMod_Reshape1")
        power = tf.tile(tf.expand_dims(tf.reduce_sum(input_mod, axis=1), 1), (1, self.K, 1)) - input_mod
        interference_f = tf.multiply(power, x)
        # up = tf.multiply(input_mod, x)
        # interference_f = tf.tile(tf.reduce_sum(up, axis=1, keepdims=True), (1, self.K, 1)) - up
        interference_f = tf.reduce_sum(interference_f, axis=2, keepdims=True)
        interference_f = tf.matmul(self.Mk, interference_f)
        unflattened_output_0 = tf.transpose(x, perm=[0, 2, 1])
        interference_t = tf.matmul(input_mod, unflattened_output_0)
        interference_t = tf.reduce_sum(interference_t - tf.multiply(interference_t, tf.eye(self.K)), axis=2)
        interference_t = tf.tile(tf.expand_dims(interference_t, axis=2), (1, self.top_l, 1))
        G_mean = tf.reduce_mean(tf.keras.layers.Reshape((self.M*self.K, ))(input_mod), axis=1, keepdims=True)
        G_mean = tf.tile(tf.expand_dims(G_mean, axis=1), (1, self.K * self.top_l, 1))
        G_max = tf.reduce_max(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
        G_max = tf.tile(tf.expand_dims(G_max, axis=1), (1, self.K * self.top_l, 1))
        G_min = tf.reduce_min(tf.keras.layers.Reshape((self.M * self.K,))(input_mod), axis=1, keepdims=True)
        G_min = tf.tile(tf.expand_dims(G_min, axis=1), (1, self.K * self.top_l, 1))
        iteration_num = tf.multiply(tf.constant(0.0), input_reshaper(smol_input_mod)) + tf.constant(step)
        input_i = input_concatnator(
            [-input_reshaper(smol_input_mod),
             G_mean, G_max, G_min,
             interference_t, interference_f,
             tf.tile(tf.expand_dims(smol_x, axis=1), (1, self.K*self.top_l, 1)),
             iteration_num])
        return input_i

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'k': self.k,
            'top_l': self.top_l,
            'name': "TopPrecoderPerUserInputMod",
            'Mk': None,
            'Mm': None
        })
        return config
# for select the top l precoders and only consider those as the output
def G_compress(G, top_l=2):
    K = G.shape[1]
    M = G.shape[2]
    G = tf.abs(G)
    compressed_G, top_indices = tf.math.top_k(G, k=top_l)
    position_matrix = np.zeros((tf.shape(G)[0], top_l * K, M))
    for n in range(0, G.shape[0]):
        for i in range(0, K * top_l):
            p_i = i % top_l
            user_i = tf.floor(i * 1.0 / top_l)
            position_matrix[n, i, int(top_indices[n, int(user_i.numpy()), p_i])] = 1.0
    position_matrix = tf.Variable(position_matrix, dtype=tf.float32, trainable=False)
    return compressed_G, position_matrix
class X_extends(tf.keras.layers.Layer):
    def __init__(self, K, M, N_rf, num, **kwargs):
        super(X_extends, self).__init__()
        self.K = K
        self.M = M
        self.N_rf = N_rf
        self.num = num # number of users considered for each choice
        self.stretch = None
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
    def call(self, x_filtered, position_matrix):
        # x_filtered = [None, K*num]
        x_filtered = tf.keras.layers.Reshape((self.K, self.num), name="X_extend_reshape_1")(x_filtered)
        if self.stretch is None:
            self.stretch = np.zeros((self.K, self.num*self.K), dtype=np.float32)
            for i in range(0, self.K):
                for j in range(0, self.num):
                    self.stretch[i, self.num*i+j] = 1.0
            self.stretch = tf.Variable(self.stretch, dtype=tf.float32, trainable=False, name="stretching_matrix")
        x_filtered = tf.tile(x_filtered, [1, 1, self.K], name="X_extend_tile_1")
        x_filtered = tf.multiply(x_filtered, self.stretch, name="X_extend_multiply_1")
        x_filtered = tf.matmul(x_filtered, position_matrix, name="X_extend_matmul_1")
        return x_filtered
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'M': self.M,
            'N_rf': self.N_rf,
            'num': self.num,
            'name': "X_extends",
        })
        return config

@tf.custom_gradient
def rebar_sm(x):
  e = tf.exp(x)
  def grad(dy):
    return dy * (1 - 1 / (1 + e))
  return tf.math.log(1 + e), grad

############################## MLP modes ##############################
def create_MLP_model(input_shape, k):
    # outputs logit
    inputs = Input(shape=input_shape)
    x = perception_model(inputs, k, 5)
    model = Model(inputs, x, name="max_nn")
    return model
def create_MLP_mean0_model(input_shape, k):
    # outputs logit
    inputs = Input(shape=input_shape)
    x = inputs - 0.5
    x = perception_model(x, k, 5)
    model = Model(inputs, x, name="max_nn")
    return model
def create_large_MLP_model(input_shape, k):
    inputs = Input(shape=input_shape)
    x = perception_model(inputs, 50, 10)
    x = LeakyReLU()(x)
    x = Dense(10)(x)
    model = Model(inputs, x, name="Deep_max_nn")
    print(model.summary())
    return model
def create_MLP_model_with_transform(input_shape, k):
    # model = create_MLP_model_with_transform((k,k), k)
    # needs to call transform first
    # outputs logit
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = perception_model(x, k, 2)
    model = Model(inputs, x, name="max_nn")
    return model
def ranking_transform(x):
    out = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for k in range(x.shape[0]):
        for i in range(0, x.shape[1]):
            for j in range(0, x.shape[1]):
                if x[k, i] >= x[k, j]:
                    out[k, i, j] = 1
    return tf.convert_to_tensor(out, dtype=tf.float32)
def create_regression_MLP_netowkr(input_shape, k):
    inputs = Input(shape=input_shape)
    x = perception_model(inputs, 1, 5)
    x = tf.squeeze(x)
    model = Model(inputs, x, name="max_nn_with_regression0")
    return model
############################## LSTM models ##############################
# model = create_LSTM_model(k, [k, 1], 10)
def create_LSTM_model(k, input_shape=[], state_size=10):
    inputs = Input(shape=input_shape)
    x = tf.keras.layers.LSTM(state_size)(inputs)
    x = LeakyReLU()(x)
    x = Dense(20)(x)
    x = LeakyReLU()(x)
    x = Dense(10)(x)
    model = Model(inputs, x, name="max_rnn")
    return model
def create_LSTM_model_backwards(k, input_shape=[]):
    inputs = Input(shape=input_shape)
    x = tf.keras.layers.LSTM(30, go_backwards=True)(inputs)
    x = LeakyReLU()(x)
    x = Dense(20)(x)
    x = LeakyReLU()(x)
    x = Dense(10)(x)
    model = Model(inputs, x, name="max_rnn")
    return model
def create_LSTM_model_with2states(k, input_shape=[], state_size=10):
    inputs = Input(shape=input_shape)
    x, final_memory_state, final_carry_state = tf.keras.layers.LSTM(30, return_sequences=True, return_state=True)(
        inputs)
    x = tf.keras.layers.Flatten()(x[:, -2:])  # looking only at the last 2 layers
    x = LeakyReLU()(x)
    x = Dense(20)(x)
    x = LeakyReLU()(x)
    x = Dense(10)(x)
    model = Model(inputs, x, name="max_rnn")
    return model
def create_BLSTM_model_with2states(k, input_shape=[], state_size=10):
    inputs = Input(shape=input_shape)
    x, final_memory_state, final_carry_state = tf.keras.layers.LSTM(30, go_backwards=True, return_sequences=True,
                                                                    return_state=True)(inputs)
    x_1 = tf.keras.layers.Flatten()(x[:, -1:])  # looking only at the last 2 layers
    x = tf.concat((tf.keras.layers.Flatten()(x[:, -1]), tf.keras.layers.Flatten()(x[:, 0])), axis=1)
    x = LeakyReLU()(x)
    x = Dense(20)(x)
    x = LeakyReLU()(x)
    x = Dense(10)(x)
    model = Model(inputs, x, name="max_rnn")
    return model
############################## Encoding models ##############################
def create_encoding_model(k, l, input_shape):
    inputs = Input(shape=input_shape)
    x_list = tf.split(inputs, num_or_size_splits=k, axis=1)
    encoding = Encoder_module(l)(x_list[0])
    for item in x_list[1:]:
        encoding = tf.concat((encoding, Encoder_module(l)(item)), axis=1)
    out = perception_model(encoding, k, 5)
    model = Model(inputs, out, name="auto_encoder_nn")
    return model
def create_uniform_encoding_model(k, l, input_shape):
    inputs = Input(shape=input_shape)
    x_list = tf.split(inputs, num_or_size_splits=k, axis=1)
    encoder_module = Uniform_Encoder_module(1, l, (1,))
    encoding = encoder_module(x_list[0])
    for item in x_list[1:]:
        encoding = tf.concat((encoding, encoder_module(item)), axis=1)
    out = perception_model(encoding, k, 5)
    model = Model(inputs, out, name="auto_encoder_nn")
    return model
def boosting_regression_model(models, input_shape, k):
    inputs = Input(shape=input_shape)
    x = models[0](inputs)
    print(x.shape)
    for model in models[1:]:
        x = tf.concat((x, model(inputs)), axis=1)
    initializer = tf.keras.initializers.Constant(1. / 2)
    x = Dense(1, kernel_initializer=initializer)(x)
    model = Model(inputs, x, name="ensemble")
    return model
def Encoder_module(L):
    def encoder_module(x):
        x = Dense(20)(x)
        x = LeakyReLU()(x)
        x = Dense(L)(x)
        # x = tf.keras.activations.tanh(x) + tf.stop_gradient(tf.math.sign(x) - tf.keras.activations.tanh(x))
        # x = sign_relu_STE(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = hard_tanh(x) + tf.stop_gradient(tf.sign(x) - hard_tanh(x))
        return x

    return encoder_module
def Uniform_Encoder_module(k, l, input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(20)(inputs)
    x = LeakyReLU()(x)
    x = Dense(l)(x)
    x = tf.keras.activations.sigmoid(x) + tf.stop_gradient(tf.math.sign(x) - tf.keras.activations.sigmoid(x))
    model = Model(inputs, x, name="encoder_unit")
    return model
def create_encoding_model_with_annealing(k, l, input_shape):
    inputs = Input(shape=input_shape)
    epoch = inputs[:, 1][0]
    inputs_mod = inputs[:, 1:]
    print(inputs.shape)
    x_list = tf.split(inputs_mod, num_or_size_splits=k, axis=1)
    encoding = Encoder_module_annealing(l)(x_list[0], epoch)
    for item in x_list[1:]:
        encoding = tf.concat((encoding, Encoder_module_annealing(l)(item, epoch)), axis=1)
    out = perception_model(encoding, k, 5)
    model = Model(inputs, out, name="k2_L2_annealing_nn")
    print(model.summary())
    return model
def Uniform_Encoder_module_with_annealing(k, l, input_shape):
    inputs = Input(shape=input_shape)
    epoch = inputs[:, 1][0]
    inputs_mod = inputs[:, 1:]
    x = Dense(20)(inputs_mod)
    x = LeakyReLU()(x)
    x = Dense(l)(x)
    x = annealing_tanh(x, epoch) + tf.stop_gradient(tf.math.sign(x) - annealing_tanh(x, epoch))
    model = Model(inputs, x, name="encoder_unit")
    return model
def Encoder_module_annealing(L):
    def encoder_module(x, N):
        x = Dense(50)(x)
        x = LeakyReLU()(x)
        x = Dense(20)(x)
        x = LeakyReLU()(x)
        x = Dense(L)(x)
        # x = sign_relu_STE(x)
        x = annealing_tanh(x, N) + tf.stop_gradient(tf.sign(x) - annealing_tanh(x, N))
        # x = hard_tanh(x) + tf.stop_gradient(tf.sign(x) - hard_tanh(x))
        return x

    return encoder_module
def create_encoding_model_with_annealing_LSTM(k, l, input_shape):
    inputs = Input(shape=input_shape)
    epoch = inputs[:, 1][0]
    inputs_mod = inputs[:, 1:]
    x_list = tf.split(inputs_mod, num_or_size_splits=k, axis=1)
    encoding = Encoder_module_annealing(l)(x_list[0], epoch)
    for item in x_list[1:]:
        encoding = tf.concat((encoding, Encoder_module_annealing(l)(item, epoch)), axis=1)
    out = create_LSTM_model_with2states(k, [k, 1], 10)
    model = Model(inputs, out, name="k2_L2_annealing_nn_LSTM")
    print(model.summary())
    return model
def ensemble_regression(k, input_shape):
    regression_1 = create_regression_MLP_netowkr(input_shape, k)
    regression_2 = tf.keras.models.load_model("trained_models/Sept 19th/N_10000_5_Layer_MLP_regression.h5")
def create_uniform_encoding_model_with_annealing(k, l, input_shape):
    inputs = Input(shape=input_shape)
    x_list = tf.split(inputs, num_or_size_splits=k + 1, axis=1)
    encoder_module = Uniform_Encoder_module_with_annealing(1, l, (2,))
    x_mod = tf.concat((x_list[0], x_list[1]), axis=1)
    encoding = encoder_module(x_mod)
    for item in x_list[2:]:
        x_mod = tf.concat((x_list[0], item), axis=1)
        encoding = tf.concat((encoding, encoder_module(x_mod)), axis=1)
    out = perception_model(encoding, k, 5)
    model = Model(inputs, out, name="auto_encoder_nn")
    return model
def perception_model(x, output, layer, logit=True):
    for i in range(layer - 1):
        x = Dense(50)(x)
        x = LeakyReLU()(x)
    x = Dense(output)(x)
    if logit:
        return x
    else:
        return Softmax(x)
def Autoencoder_Encoding_module(input_shape, i=0, code_size=15, normalization=False):
    inputs = Input(input_shape, dtype=tf.float32)
    if normalization:
        min = tf.tile(tf.expand_dims(tf.reduce_min(inputs, axis=2), axis=2), (1, 1, inputs.shape[2]))
        max = tf.tile(tf.expand_dims(tf.reduce_max(inputs, axis=2), axis=2), (1, 1, inputs.shape[2]))
        x = (inputs - min) / (max - min)
    else:
        x = inputs
    x = Dense(512, kernel_initializer=tf.keras.initializers.he_normal(), name="encoder_{}_dense_1".format(i))(x)
    x = LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Dense(code_size, kernel_initializer=tf.keras.initializers.he_normal(), name="encoder_{}_dense_2".format(i))(x)
    return Model(inputs, x, name="encoder_{}".format(i))
def Autoencoder_Encoding_module_sig(input_shape, i=0, code_size=15, normalization=False):
    inputs = Input(input_shape, dtype=tf.float32)
    if normalization:
        min = tf.tile(tf.expand_dims(tf.reduce_min(inputs, axis=2), axis=2), (1, 1, inputs.shape[2]))
        max = tf.tile(tf.expand_dims(tf.reduce_max(inputs, axis=2), axis=2), (1, 1, inputs.shape[2]))
        x = (inputs - min) / (max - min)
    else:
        x = inputs
    x = Dense(512, kernel_initializer=tf.keras.initializers.he_normal(), name="encoder_{}_dense_1".format(i))(x)
    x = sigmoid(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Dense(code_size, kernel_initializer=tf.keras.initializers.he_normal(), name="encoder_{}_dense_2".format(i))(x)
    return Model(inputs, x, name="encoder_{}".format(i))
def Autoencoder_cnn_Encoding_module(input_shape, i=0, code_size=4):
    inputs = Input(input_shape, dtype=tf.float32)
    K = input_shape[0]
    M = input_shape[1]
    distribute = tf.keras.layers.TimeDistributed
    inputs_mod = tf.keras.layers.Reshape((K, M, 1))(inputs) # N, K, M, 1
    x = distribute(tf.keras.layers.Conv1D(16, 4, 4))(inputs_mod)
    x = distribute(Dense(64))(x)
    x = LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = distribute(Dense(code_size))(x)
    x = tf.keras.layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)
    return Model(inputs, x, name="encoder_{}".format(i))
def Autoencoder_chunky_Encoding_module(input_shape, i=0, code_size=4, splits=16):
    inputs = Input(input_shape, dtype=tf.float32)
    K = input_shape[0]
    M = input_shape[1]
    distribute = tf.keras.layers.TimeDistributed
    input_mod = tf.keras.layers.Reshape((K, int(splits), int(M/splits)))(inputs)
    x = distribute(Dense(512))(input_mod)
    x = LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = distribute(Dense(code_size))(x)
    x = tf.keras.layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)
    return Model(inputs, x, name="encoder_{}".format(i))
def Autoencoder_chunky_Decoding_module(input_shape, i=0, M=64, splits=16):
    inputs = Input(input_shape, dtype=tf.float32)
    K = input_shape[0]
    code_size = int(input_shape[1]/splits)
    distribute = tf.keras.layers.TimeDistributed
    inputs_mod = tf.keras.layers.Reshape((K, splits, code_size))(inputs) # N, K, M, 1
    x = distribute(Dense(512))(inputs_mod)
    x = LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = distribute(Dense(M/splits))(x)
    x = tf.keras.layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)
    return Model(inputs, x, name="decoder_{}".format(i))
def Autoencoder_layers_Encoding_module(input_shape, i=0, max_bits=4, splits=16):
    inputs = Input(input_shape, dtype=tf.float32)
    x_1 = []
    for i in range(splits):
        layer_i = Dense(256)
        x = layer_i(inputs)
        x = LeakyReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = Dense(max_bits)(x)
        x_1.append(x)
    x = tf.concat(x_1, axis=2)
    return Model(inputs, x, name="encoder_{}".format(i))
def Autoencoder_layers_Decoding_module(input_shape, i=0, M=64, splits=16):
    inputs = Input(input_shape, dtype=tf.float32)
    K = input_shape[0]
    max_code_size = int(input_shape[1]/splits)
    inputs_mod = tf.keras.layers.Reshape((K, splits, max_code_size))(inputs) # N,
    out = []
    for i in range(splits):
        x = Dense(256)(inputs_mod[:, :, i])
        x = LeakyReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = Dense(M)(x)
        out.append(x)
    x = tf.add_n(out)
    return Model(inputs, x, name="decoder_{}".format(i))
def Autoencoder_rates_Encoding_module(input_shape, i=0, rates=[], splits=4):
    inputs = Input(input_shape, dtype=tf.float32)
    x_1 = []
    for i in range(splits):
        layer_i = Dense(8*rates[i])
        x = layer_i(inputs)
        x = LeakyReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = Dense(rates[i])(x)
        x_1.append(x)
    x = tf.concat(x_1, axis=2)
    return Model(inputs, x, name="encoder_{}".format(i))
def Autoencoder_rates_Decoding_module(input_shape, i=0, M=64, rates=[], splits=4):
    inputs = Input(input_shape, dtype=tf.float32)
    K = input_shape[0]
    out = []
    for i in range(splits):
        if i == 0:
            current = 0
        else:
            current = tf.add_n(rates[:i])
        x = Dense(8*rates[i])(inputs[:, :, current:current+rates[i]])
        x = LeakyReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = Dense(M)(x)
        out.append(x)
    x = tf.add_n(out)
    return Model(inputs, x, name="decoder_{}".format(i))
def Autoencoder_Decoding_module(output_size, input_shape, i=0):
    inputs = Input(input_shape)
    x = Dense(512, kernel_initializer=tf.keras.initializers.he_normal(), name="decoder_{}_dense_1".format(i))(inputs)
    x = LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Dense(output_size, kernel_initializer=tf.keras.initializers.he_normal(), name="decoder_{}_dense_2".format(i))(x)
    return Model(inputs, x, name="decoder_{}".format(i))
def Autoencoder_Decoding_module_sig(output_size, input_shape, i=0):
    inputs = Input(input_shape)
    x = Dense(512, kernel_initializer=tf.keras.initializers.he_normal(), name="decoder_{}_dense_1".format(i))(inputs)
    x = sigmoid(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Dense(output_size, kernel_initializer=tf.keras.initializers.he_normal(), name="decoder_{}_dense_2".format(i))(x)
    return Model(inputs, x, name="decoder_{}".format(i))
def DiscreteVAE(k, l, input_shape, code_size=15):
    inputs = Input(input_shape, dtype=tf.float32)
    x_list = tf.split(inputs, num_or_size_splits=k, axis=1)
    # list of modules
    encoder = Autoencoder_Encoding_module(k, l, (1,), code_size=code_size)
    decoder = Autoencoder_Decoding_module(k, code_size, (k * code_size,))
    find_nearest_e = Closest_embedding_layer(user_count=k, embedding_count=2 ** l, bit_count=code_size, i=0)
    encoding_reshaper = tf.keras.layers.Reshape((k * code_size,), name="encoding_reshaper")
    # computation of encoding
    z_e = encoder(x_list[0])
    i = 1
    for item in x_list[1:]:
        # z_e = tf.concat((z_e, Autoencoder_Encoding_module(k, l, (1, ), i)(item)), axis=1)
        z_e = tf.concat((z_e, encoder(item)), axis=1)
        i = i + 1
    print(z_e.shape)
    z_qq = find_nearest_e(z_e)
    z_fed_forward = z_e + tf.stop_gradient(z_qq - z_e)
    z_fed_forward = encoding_reshaper(z_fed_forward)
    z_e = encoding_reshaper(z_e)
    z_qq = encoding_reshaper(z_qq)
    out = decoder(z_fed_forward)
    output_all = tf.keras.layers.concatenate((out, z_qq, z_e), 1)
    model = Model(inputs, output_all, name="DiscreteVAE")
    return model
def DiscreteVAE_diff_scheduler(k, l, input_shape, code_size=15):
    inputs = Input(input_shape, dtype=tf.float32)
    x_list = tf.split(inputs, num_or_size_splits=k, axis=1)
    # list of modules
    decoder = Autoencoder_Decoding_module(k, code_size, (k * code_size,))
    encoding_reshaper = tf.keras.layers.Reshape((k * code_size,), name="encoding_reshaper")
    # computation of encoding
    # z_e = Autoencoder_Encoding_module(k, l, (1, ))(x_list[0])
    z_e = Autoencoder_Encoding_module(k, l, (1,), code_size=code_size)(x_list[0])
    z_q = Closest_embedding_layer(user_count=k, embedding_count=2 ** l, bit_count=code_size)(z_e)
    z_fed_forward = z_e + tf.stop_gradient(z_q - z_e)
    i = 1
    for item in x_list[1:]:
        # z_e = tf.concat((z_e, Autoencoder_Encoding_module(k, l, (1, ), i)(item)), axis=1)
        z_e_temp = Autoencoder_Encoding_module(k, l, (1,), i=i, code_size=code_size)(item)
        z_q_temp = Closest_embedding_layer(user_count=k, embedding_count=2 ** l, bit_count=code_size)(z_e_temp)
        z_fed_forward_temp = z_e_temp + tf.stop_gradient(z_q_temp - z_e_temp)
        z_e = tf.concat((z_e, z_e_temp), axis=1)
        z_q = tf.concat((z_q, z_q_temp), axis=1)
        z_fed_forward = tf.concat((z_fed_forward, z_fed_forward_temp), axis=1)
        i = i + 1
    z_fed_forward = encoding_reshaper(z_fed_forward)
    z_e = encoding_reshaper(z_e)
    z_q = encoding_reshaper(z_q)
    out = decoder(z_fed_forward)
    output_all = tf.keras.layers.concatenate((out, z_q, z_e), 1)
    model = Model(inputs, output_all, name="DiscreteVAE")
    return model
def DiscreteVAE_regression(l, input_shape, code_size=15):
    inputs = Input(input_shape, dtype=tf.float32)
    encoder = Autoencoder_Encoding_module(3, l, (1,), code_size=code_size)
    decoder = Autoencoder_Decoding_module(1, code_size, (code_size,))
    find_nearest_e = Closest_embedding_layer(1, 2 ** l, code_size)
    encoding_reshaper = tf.keras.layers.Reshape((code_size,), name="encoding_reshaper")
    z_e = encoder(inputs)
    z_qq = find_nearest_e(z_e)
    z_fed_forward = z_e + tf.stop_gradient(z_qq - z_e)
    z_fed_forward = encoding_reshaper(z_fed_forward)
    z_e = encoding_reshaper(z_e)
    z_qq = encoding_reshaper(z_qq)
    out = decoder(z_fed_forward)
    output_all = tf.keras.layers.concatenate((out, z_qq, z_e), 1)
    model = Model(inputs, output_all, name="DiscreteVAEregression")
    print(model.summary())
    return model
############################## Encoding models with bitstring input ##############################
def F_Encoder_module_annealing(L, i=0):
    def encoder_module(x, N):
        x = Dense(7, name="encoder_dense_1_{}".format(i))(x)
        x = LeakyReLU()(x)
        x = Dense(5, name="encoder_dense_4_{}".format(i))(x)
        x = LeakyReLU()(x)
        x = Dense(5, name="encoder_dense_2_{}".format(i))(x)
        x = LeakyReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = Dense(L, name="encoder_dense_3_{}".format(i))(x)
        # x = annealing_tanh(x, N, name="tanh_pos_{}".format(i)) + \
        #     tf.stop_gradient(tf.math.sign(x, name="encoder_sign_{}".format(i)) - annealing_tanh(x, N, name="tanh_neg_{}".format(i)))
        x = tf.tanh(tf.keras.layers.ReLU()(x), name="tanh_pos_{}".format(i)) + tf.stop_gradient(
            binary_activation(x) - tf.tanh(tf.keras.layers.ReLU()(x), name="tanh_neg_{}".format(i)))
        # x = annealing_tanh(tf.keras.layers.ReLU()(x), N, name="tanh_pos_{}".format(i)) + tf.stop_gradient(
        #     binary_activation(x) - annealing_tanh(tf.keras.layers.ReLU()(x), N, name="tanh_neg_{}".format(i)))
        return x

    return encoder_module
def F_create_encoding_model_with_annealing(k, l, input_shape):
    # features_mod = tf.ones((features.shape[1], 1)) * N
    # features_mod = tf.concat((features_mod, features), axis=2)
    # F_create_encoding_model_with_annealing(2, 1, (2, 24))
    inputs = Input(shape=input_shape)
    epoch = inputs[0, 0, 0]
    inputs_mod = inputs[:, :, 1:]
    x_list = tf.split(inputs_mod, num_or_size_splits=k, axis=1)
    encoding = F_Encoder_module_annealing(l, 0)(x_list[0][:, 0, :], epoch)
    for i in range(1, len(x_list)):
        encoding = tf.concat((encoding, F_Encoder_module_annealing(l, i)(x_list[i][:, 0, :], epoch)), axis=1)
    x = tf.keras.layers.Dense(20)(encoding)
    x = sigmoid(x)
    out = tf.keras.layers.Dense(k)(x)
    model = Model(inputs, out, name="k2_L2_annealing_nn_on_floatpoint_bit")
    print(model.summary())
    return model
def F_create_LSTM_encoding_model_with_annealing(k, l, input_shape):
    # features_mod = tf.ones((features.shape[1], 1)) * N
    # features_mod = tf.concat((features_mod, features), axis=2)
    # F_create_encoding_model_with_annealing(2, 1, (2, 24))
    inputs = Input(shape=input_shape)
    epoch = inputs[0, 0, 0]
    inputs_mod = inputs[:, :, 1:]
    x_list = tf.split(inputs_mod, num_or_size_splits=k, axis=1)
    encoding = F_LSTM_Encoder_module_annealing(l, 0)(x_list[0][:, 0, :], epoch)
    for i in range(1, len(x_list)):
        encoding = tf.concat((encoding, F_LSTM_Encoder_module_annealing(l, i)(x_list[i][:, 0, :], epoch)), axis=1)
    x = tf.keras.layers.Dense(20)(encoding)
    x = sigmoid(x)
    out = tf.keras.layers.Dense(2)(x)
    model = Model(inputs, out, name="k2_L2_annealing_nn_on_floatpoint_bit_CNN")
    print(model.summary())
    return model
def F_LSTM_Encoder_module_annealing(L, i=0):
    def encoder_module(x, N):
        x = tf.keras.layers.Reshape((23, 1))(x)
        x = tf.keras.layers.LSTM(4, name="LSTM_{}".format(i), kernel_initializer=tf.keras.initializers.he_uniform())(x)
        x = Dense(5, name="encoder_dense_1_{}".format(i), kernel_initializer=tf.keras.initializers.he_uniform())(x)
        x = LeakyReLU()(x)
        x = Dense(5, name="encoder_dense_2_{}".format(i), kernel_initializer=tf.keras.initializers.he_uniform())(x)
        x = LeakyReLU()(x)
        x = Dense(L, name="encoder_dense_3_{}".format(i), kernel_initializer=tf.keras.initializers.he_uniform())(x)
        # x = leaky_hard_tanh(tf.keras.layers.ReLU()(x)) + tf.stop_gradient(
        #     binary_activation(x) - leaky_hard_tanh(tf.keras.layers.ReLU()(x)))
        # x = annealing_tanh(x, N, name="tanh_pos_{}".format(i)) + \
        #     tf.stop_gradient(tf.math.sign(x, name="encoder_sign_{}".format(i)) - annealing_tanh(x, N, name="tanh_neg_{}".format(i)))
        x = tf.tanh(tf.keras.layers.ReLU()(x), name="tanh_pos_{}".format(i)) + tf.stop_gradient(
            binary_activation(x) - tf.tanh(tf.keras.layers.ReLU()(x), name="tanh_neg_{}".format(i)))
        # x = annealing_tanh(tf.keras.layers.ReLU()(x), N, name="tanh_pos_{}".format(i)) + tf.stop_gradient(
        #     binary_activation(x) - annealing_tanh(tf.keras.layers.ReLU()(x), N, name="tanh_neg_{}".format(i)))
        return x

    return encoder_module
def F_create_CNN_encoding_model_with_annealing(k, l, input_shape):
    # features_mod = tf.ones((features.shape[1], 1)) * N
    # features_mod = tf.concat((features_mod, features), axis=2)
    # F_create_encoding_model_with_annealing(2, 1, (2, 24))
    inputs = Input(shape=input_shape)
    epoch = inputs[0, 0, 0]
    inputs_mod = inputs[:, :, 1:]
    x_list = tf.split(inputs_mod, num_or_size_splits=k, axis=1)
    encoding = F_CNN_Encoder_module_annealing(l, 0)(x_list[0][:, 0, :], epoch)
    for i in range(1, len(x_list)):
        encoding = tf.concat((encoding, F_CNN_Encoder_module_annealing(l, i)(x_list[i][:, 0, :], epoch)), axis=1)
    x = tf.keras.layers.Dense(20)(encoding)
    x = sigmoid(x)
    out = tf.keras.layers.Dense(2)(x)
    model = Model(inputs, out, name="k2_L2_annealing_nn_on_floatpoint_bit_CNN")
    print(model.summary())
    return model
def F_CNN_Encoder_module_annealing(L, i=0):
    def encoder_module(x, N):
        x = tf.keras.layers.Reshape((23, 1))(x)
        # x = tf.keras.layers.LSTM(8, name="LSTM_{}".format(i), kernel_initializer=tf.keras.initializers.he_normal())(x)
        x = tf.keras.layers.Conv1D(filters=10, kernel_size=5, strides=1)(x)
        x = tf.keras.layers.Reshape((70,))(x)
        x = Dense(5, name="encoder_dense_1_{}".format(i))(x)
        x = LeakyReLU()(x)
        x = Dense(5, name="encoder_dense_2_{}".format(i))(x)
        x = LeakyReLU()(x)
        x = Dense(5, name="encoder_dense_3_{}".format(i))(x)
        x = LeakyReLU()(x)
        x = Dense(L, name="encoder_dense_4_{}".format(i))(x)
        # x = annealing_tanh(x, N, name="tanh_pos_{}".format(i)) + \
        #     tf.stop_gradient(tf.math.sign(x, name="encoder_sign_{}".format(i)) - annealing_tanh(x, N, name="tanh_neg_{}".format(i)))
        x = tf.tanh(tf.keras.layers.ReLU()(x), name="tanh_pos_{}".format(i)) + tf.stop_gradient(
            binary_activation(x) - tf.tanh(tf.keras.layers.ReLU()(x), name="tanh_neg_{}".format(i)))
        # x = annealing_tanh(tf.keras.layers.ReLU()(x), N, name="tanh_pos_{}".format(i)) + tf.stop_gradient(
        #     binary_activation(x) - annealing_tanh(tf.keras.layers.ReLU()(x), N, name="tanh_neg_{}".format(i)))
        return x

    return encoder_module
def Thresholdin_network(input_shape):
    inputs = Input(shape=input_shape)
    epoch = inputs[0, 0, 0]
    inputs_mod = inputs[:, :, 1:]
    x = SubtractLayer_with_noise(name="threshold")(inputs_mod)
    x = tf.keras.layers.Reshape((2,))(x)
    x = tf.tanh(tf.keras.layers.ReLU()(x), name="tanh_pos") + tf.stop_gradient(
        binary_activation(x) - tf.tanh(tf.keras.layers.ReLU()(x), name="tanh_neg"))
    x = Dense(20, name="decoder_dense_1")(x)
    x = LeakyReLU()(x)
    x = Dense(2, name="decoder_dense_2")(x)
    model = Model(inputs, x)
    print(model.summary())
    print(model.trainable_variables)
    return model
def F_swapadoo_Encoder(k, l, input_shape):
    inputs = Input(shape=input_shape)
############################## Encoding models with Prof Yu's proposal input ##############################
def F_create_encoding_regression_module(input_shape, levels, j=0):
    inputs = Input(shape=input_shape)
    x = Dense(8, name="encoding_dense_1_" + str(j))(inputs)
    x = LeakyReLU()(x)
    x = Dense(5, name="encoding_dense_2_" + str(j))(x)
    x = LeakyReLU()(x)
    x = Dense(5, name="encoding_dense_3_" + str(j))(x)
    x = LeakyReLU()(x)
    x = Dense(1, name="encoding_output_real_" + str(j))(x)
    x = hard_sigmoid(x)
    x = hard_sigmoid(x) + tf.stop_gradient(multi_level_thresholding(x, levels) - hard_sigmoid(x))
    # x = annealing_tanh(x, epoch) + tf.stop_gradient(tf.math.sign(x) - annealing_tanh(x, epoch))
    model = Model(inputs, x, name="encoder_unit_" + str(j))
    return model
def F_creating_common_encoding_regression(input_shape, levels=2, k=2):
    inputs = Input(shape=input_shape)
    epoch = inputs[0, 0, 0]
    inputs_mod = inputs[:, :, 1:]
    x_list = tf.split(inputs_mod, num_or_size_splits=k, axis=1)
    encoding_model = F_create_encoding_regression_module((inputs_mod.shape[2]), levels)
    encoding = encoding_model(x_list[0][:, 0, :])
    for i in range(1, len(x_list)):
        encoding = tf.concat((encoding, encoding_model(x_list[i][:, 0, :])), axis=1)
    model = Model(inputs, encoding, name="encoder_network")
    return model
def F_creating_distinct_encoding_regression(input_shape, levels=2, k=2):
    inputs = Input(shape=input_shape)
    epoch = inputs[0, 0, 0]
    inputs_mod = inputs[:, :, 1:]
    x_list = tf.split(inputs_mod, num_or_size_splits=k, axis=1)
    encoding = F_create_encoding_regression_module((inputs_mod.shape[2]), levels)(x_list[0][:, 0, :])
    for i in range(1, len(x_list)):
        encoding = tf.concat(
            (encoding, F_create_encoding_regression_module((inputs_mod.shape[2]), levels, j=i)(x_list[i][:, 0, :])),
            axis=1)
    model = Model(inputs, encoding, name="encoder_network")
    return model
############################## FDD Scheduling Models ##############################
def CommonFDD_Quantizer(M, B, K, i=0):
    inputs = Input(shape=[M, ])
    x = Dense(M * K)(inputs)
    x = LeakyReLU()(x)
    x = Dense(M * K)(x)
    x = LeakyReLU()(x)
    x = Dense(K)(x)
    x = LeakyReLU()(x)
    x = Dense(B)(x)
    x = tf.tanh(tf.keras.layers.ReLU()(x), name="tanh_pos_{}".format(i)) + tf.stop_gradient(
        binary_activation(x) - tf.tanh(tf.keras.layers.ReLU()(x), name="tanh_neg_{}".format(i)))
    model = Model(inputs, x, name="commonFDD_quantizer")
    return model
def FDD_encoding_model_no_constraint(M, K, B):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    x = tf.keras.layers.Concatenate(axis=2)([tf.math.real(inputs), tf.math.imag(inputs)])
    quantizer = CommonFDD_Quantizer(2 * M, B, K)
    x_list = tf.split(x, num_or_size_splits=K, axis=1)
    reshaper = tf.keras.layers.Reshape((2 * M,))
    encoding = quantizer(reshaper(x_list[0]))
    for i in range(1, len(x_list)):
        encoding = tf.concat((encoding, quantizer(reshaper(x_list[i]))), axis=1)
    x = Dense(2 * M * K)(encoding)
    x = LeakyReLU()(x)
    x = Dense(M * K)(x)
    # to be removed
    # output = tf.tanh(tf.keras.layers.ReLU()(x))
    output = sigmoid(x)
    model = Model(inputs, output)
    return model
def Floatbits_FDD_encoding_model_no_constraint(M, K, B):
    inputs = Input(shape=(K, M * 2 * 23), dtype=tf.float32)
    quantizer = CommonFDD_Quantizer(M * 2 * 23, B, K)
    x_list = tf.split(inputs, num_or_size_splits=K, axis=1)
    reshaper = tf.keras.layers.Reshape((2 * M * 23,))
    encoding = quantizer(reshaper(x_list[0]))
    for i in range(1, len(x_list)):
        encoding = tf.concat((encoding, quantizer(reshaper(x_list[i]))), axis=1)
    x = Dense(2 * M * K)(encoding)
    x = LeakyReLU()(x)
    x = Dense(M * K)(x)
    # to be removed
    # output = tf.tanh(tf.keras.layers.ReLU()(x))
    output = sigmoid(x)
    model = Model(inputs, output)
    return model
def FDD_encoding_model_constraint_13_with_softmax(M, K, B):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    x = tf.keras.layers.Concatenate(axis=2)([tf.math.real(inputs), tf.math.imag(inputs)])
    quantizer = CommonFDD_Quantizer(2 * M, B, K)
    x_list = tf.split(x, num_or_size_splits=K, axis=1)
    reshaper = tf.keras.layers.Reshape((2 * M,))
    encoding = quantizer(reshaper(x_list[0]))
    for i in range(1, len(x_list)):
        encoding = tf.concat((encoding, quantizer(reshaper(x_list[i]))), axis=1)
    x = Dense(M ** 2)(encoding)
    x = LeakyReLU()(x)
    x = Dense(M * K)(encoding)
    x = LeakyReLU()(x)
    x = Dense(M * K)(x)
    # to be removed
    x_list2 = tf.split(x, num_or_size_splits=K, axis=1)
    output = tf.keras.layers.Softmax()(x_list2[0])
    for i in range(1, len(x_list2)):
        output = tf.concat((output, tf.keras.layers.Softmax()(x_list2[i])), axis=1)
    # yep
    # x = tf.sigmoid(x)
    model = Model(inputs, output)
    return model
def Floatbits_FDD_encoding_model_constraint_13_with_softmax(M, K, B):
    inputs = Input(shape=(K, M * 2 * 23), dtype=tf.float32)
    quantizer = CommonFDD_Quantizer(M * 2 * 23, B, K)
    x_list = tf.split(inputs, num_or_size_splits=K, axis=1)
    reshaper = tf.keras.layers.Reshape((2 * M * 23,))
    encoding = quantizer(reshaper(x_list[0]))
    for i in range(1, len(x_list)):
        encoding = tf.concat((encoding, quantizer(reshaper(x_list[i]))), axis=1)
    x = Dense(M ** 2)(encoding)
    x = LeakyReLU()(x)
    x = Dense(M * K)(encoding)
    x = LeakyReLU()(x)
    x = Dense(M * K)(x)
    # to be removed
    x_list2 = tf.split(x, num_or_size_splits=K, axis=1)
    output = tf.keras.layers.Softmax()(x_list2[0])
    for i in range(1, len(x_list2)):
        output = tf.concat((output, tf.keras.layers.Softmax()(x_list2[i])), axis=1)
    # yep
    # x = tf.sigmoid(x)
    model = Model(inputs, output)
    return model
def LSTM_Ranking_model(M, K, k, sum_all=True):
    inputs = Input(shape=[M * K, ], name="ranking_network_input")
    x_reshape = tf.expand_dims(inputs, 1)
    x = tf.tile(x_reshape, [1, k, 1])
    lstm_ranker = tf.keras.layers.LSTM(K, return_sequences=True)
    x = lstm_ranker(x)
    x = tf.nn.softmax(x, axis=2)
    if sum_all:
        x = tf.reduce_sum(x, axis=1)
    model = Model(inputs, x)
    return model
def FDD_encoding_model_constraint_123_with_softmax_and_soft_mask(M, K, B, k=3):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    x = tf.keras.layers.Concatenate(axis=2)([tf.math.real(inputs), tf.math.imag(inputs)])
    quantizer = CommonFDD_Quantizer(2 * M, B, K)
    x_list = tf.split(x, num_or_size_splits=K, axis=1)
    reshaper = tf.keras.layers.Reshape((2 * M,))
    encoding = quantizer(reshaper(x_list[0]))
    for i in range(1, len(x_list)):
        encoding = tf.concat((encoding, quantizer(reshaper(x_list[i]))), axis=1)
    x = Dense(2 * M * K, name="start_of_decoding")(encoding)
    x = LeakyReLU()(x)
    x = Dense(2 * M * K)(x)
    x = LeakyReLU()(x)
    x = Dense(M * K)(x)
    ranking_output = LSTM_Ranking_model(M, K, k)(x)
    # ranking_output = Dense(K*M)(x)
    # ranking_output = Dense(K)(ranking_output)
    # ranking_output = tf.tanh(LeakyReLU()(ranking_output))
    # perform softmax to all the results
    x_list2 = tf.split(x, num_or_size_splits=K, axis=1)
    output = tf.keras.layers.Softmax()(x_list2[0])
    for i in range(1, len(x_list2)):
        output = tf.concat((output, tf.keras.layers.Softmax()(x_list2[i])), axis=1)
    # yep
    # x = tf.sigmoid(x)
    output = tf.concat((output, ranking_output), axis=1)
    model = Model(inputs, output)
    return model
def Floatbits_FDD_encoding_model_constraint_123_with_softmax_and_soft_mask(M, K, B, k=3):
    inputs = Input(shape=(K, M * 2 * 23), dtype=tf.float32)
    quantizer = CommonFDD_Quantizer(M * 2 * 23, B, K)
    x_list = tf.split(inputs, num_or_size_splits=K, axis=1)
    reshaper = tf.keras.layers.Reshape((2 * M * 23,))
    encoding = quantizer(reshaper(x_list[0]))
    for i in range(1, len(x_list)):
        encoding = tf.concat((encoding, quantizer(reshaper(x_list[i]))), axis=1)
    x = Dense(2 * M * K, name="start_of_decoding")(encoding)
    x = LeakyReLU()(x)
    x = Dense(2 * M * K)(x)
    x = LeakyReLU()(x)
    x = Dense(M * K)(x)
    ranking_output = LSTM_Ranking_model(M, K, k)(x)
    # ranking_output = Dense(K*M)(x)
    # ranking_output = Dense(K)(ranking_output)
    # ranking_output = tf.tanh(LeakyReLU()(ranking_output))
    # perform softmax to all the results
    x_list2 = tf.split(x, num_or_size_splits=K, axis=1)
    output = tf.keras.layers.Softmax()(x_list2[0])
    for i in range(1, len(x_list2)):
        output = tf.concat((output, tf.keras.layers.Softmax()(x_list2[i])), axis=1)
    # yep
    # x = tf.sigmoid(x)
    output = tf.concat((output, ranking_output), axis=1)
    model = Model(inputs, output)
    return model
def FDD_encoding_model_constraint_13_with_regularization(M, K, B):
    inputs = Input(shape=(K, M), dtype=tf.complex128)
    x = tf.keras.layers.Concatenate(axis=2)([tf.math.real(inputs), tf.math.imag(inputs)])
    quantizer = CommonFDD_Quantizer(2 * M, B, K)
    x_list = tf.split(x, num_or_size_splits=K, axis=1)
    reshaper = tf.keras.layers.Reshape((2 * M,))
    encoding = quantizer(reshaper(x_list[0]))
    for i in range(1, len(x_list)):
        encoding = tf.concat((encoding, quantizer(reshaper(x_list[i]))), axis=1)
    x = Dense(2 * M * K)(encoding)
    x = LeakyReLU()(x)
    x = Dense(M * K)(x)
    x = tf.tanh(tf.keras.layers.ReLU()(x), name="tanh_neg_output") + tf.stop_gradient(
        binary_activation(x) - tf.tanh(tf.keras.layers.ReLU()(x), name="tanh_neg_output"))
    model = Model(inputs, x)
    return model
############################## FDD Scheduling Models No quantizing ##############################
def FDD_model_no_constraint(M, K, B):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    x = tf.keras.layers.Concatenate(axis=2)([tf.math.real(inputs), tf.math.imag(inputs)])
    # create input vector
    reshaper = tf.keras.layers.Reshape((2 * M * K,))
    x = reshaper(x)
    x = Dense(3 * M, kernel_initializer=tf.keras.initializers.he_normal())(x)
    x = LeakyReLU()(x)
    x = Dense(M, kernel_initializer=tf.keras.initializers.he_normal())(x)
    x = LeakyReLU()(x)
    x = Dense(M * K, kernel_initializer=tf.keras.initializers.he_normal())(x)
    # to be removed
    # output = tf.tanh(tf.keras.layers.ReLU()(x))
    # output = leaky_hard_sigmoid(x)
    output = sigmoid(x)
    model = Model(inputs, output)
    return model
# def Fully_connected_Ranking_Model(M, K, k):
#     inputs = Input(shape=(M*K), dtype=tf.float32)
#     x = Dense(M*K)
#     x = Dense(M*K)
def DNN_Ranking_NN_submodule(M, K, k, inputshape):
    inputs = Input(inputshape)
    x = Dense(M * K)(inputs)
    x = Dense(2 * M)(x)
    x = Dense(M * K)(x)
    x = Dense(K)(x)
    x = tf.keras.layers.Softmax()(x)
    dnn_model = Model(inputs, x, name="dnn_softmax_network")
    return dnn_model
def DNN_Ranking_model(input_shape, M, K, k, sum_all=False):
    inputs = Input(input_shape)
    dnn = DNN_Ranking_NN_submodule(M, K, k, input_shape)
    stretch_matrix = np.zeros((M * K, K))
    for i in range(0, K):
        for j in range(0, M):
            stretch_matrix[i * M + j, i] = 1
    stretch_matrix = tf.constant(stretch_matrix, tf.float32)
    output = dnn(inputs)
    for i in range(1, k):
        # tiled_stretch_matrix = tf.tile(tf.expand_dims(stretch_matrix, 0), [1, 1, 1])
        stretched_rank_matrix = tf.matmul(stretch_matrix, tf.keras.layers.Reshape((output.shape[1], 1))(output))
        stretched_rank_matrix = tf.keras.layers.Reshape((stretched_rank_matrix.shape[1],))(stretched_rank_matrix)
        output = output + dnn(tf.multiply(stretched_rank_matrix, inputs))
    model = Model(inputs, output, name="dnn_ranking_module")
    return model
def FDD_with_CNN(M, K, N_rf):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.keras.layers.Concatenate(axis=2)([tf.math.real(inputs), tf.math.imag(inputs)])
    c = tf.keras.layers.Conv2D(M, (1, 3))(tf.keras.layers.Reshape((K, 2 * M, 1))(input_mod))
    c = tf.keras.layers.Reshape((c.shape[1] * c.shape[2] * c.shape[3],))(c)
    x = tf.keras.layers.Reshape((2 * K * M,))(input_mod)
    x = tf.concat((c, x), axis=1)
    x = Dense(3 * M * K)(x)
    x = LeakyReLU()(x)
    x = Dense(2 * M * K)(x)
    x = LeakyReLU()(x)
    x = Dense(M * K)(x)
    x = tf.keras.layers.Softmax()(x)
    model = Model(inputs, x)
    print(model.summary())
    return model
def Floatbits_FDD_model_no_constraint(M, K, B):
    inputs = Input(shape=(K, M * 2 * 23), dtype=tf.float32)
    # create input vector
    reshaper = tf.keras.layers.Reshape((2 * M * K * 23,))
    x = reshaper(inputs)
    x = Dense(M * M)(x)
    x = LeakyReLU()(x)
    x = Dense(M * M)(x)
    x = LeakyReLU()(x)
    x = Dense(M * K)(x)
    # to be removed
    output = tf.tanh(tf.keras.layers.ReLU()(x))
    # output = tf.keras.layers.Softmax()(x)
    # output = sigmoid(x)
    model = Model(inputs, output)
    return model
def DNN_3_layer_model(input_shape, M, K, i=0):
    inputs = Input(shape=input_shape, dtype=tf.float32)
    x = Dense(3 * M)(inputs)
    x = LeakyReLU()(x)
    x = Dense(M)(x)
    x = LeakyReLU()(x)
    x = Dense(M * K)(x)
    x = tf.keras.layers.Softmax()(x)
    model = Model(inputs, x, name="pass_{}".format(i))
    return model
def DNN_3_layer_model_sigmoid(input_shape, M, K, i=0):
    inputs = Input(shape=input_shape, dtype=tf.float32)
    size = 32
    x = Dense(size)(inputs)
    x = LeakyReLU()(x)
    x = Dense(size)(inputs)
    x = LeakyReLU()(x)
    x = Dense(M * K)(x)
    x = tf.sigmoid(x)
    model = Model(inputs, x, name="pass_{}".format(i))
    return model
def DNN_3_layer_model_harder_softmax(input_shape, M, K, i=0):
    inputs = Input(shape=input_shape, dtype=tf.float32)
    x = Dense(3 * M)(inputs)
    x = LeakyReLU()(x)
    x = Dense(M)(x)
    x = LeakyReLU()(x)
    x = Dense(M * K)(x)
    x = tf.keras.layers.Softmax()(10 * x)
    model = Model(inputs, x, name="pass_{}".format(i))
    return model
def DNN_3_layer_Thicc_model(input_shape, M, K, Nrf=3, i=0):
    inputs = Input(shape=input_shape, dtype=tf.float32)
    x = Dense(256)(inputs)
    x = LeakyReLU()(x)
    x = Dense(128)(x)
    x = LeakyReLU()(x)
    x = Dense(M * K * Nrf)(x)
    model = Model(inputs, x, name="pass_{}".format(i))
    return model
def FDD_softmax_k_times_with_magnitude(M, K, k):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.abs(inputs)
    input_mod = input_mod * 10.0
    input_mod = tf.keras.layers.Reshape((K * M,))(input_mod)
    decision_0 = tf.stop_gradient(tf.multiply(tf.zeros((K * M)), input_mod[:, :K * M]))
    input_pass_0 = tf.keras.layers.Concatenate(axis=1)((decision_0, input_mod))
    # dnn_model = DNN_3_layer_model((3*K*M), M, K, 0)
    x = DNN_3_layer_Thicc_model((2 * K * M), M, K, 0)(input_pass_0)
    for i in range(1, k):
        decision_i = x
        input_pass_i = tf.keras.layers.Concatenate(axis=1)((decision_i, input_mod))
        x = x + DNN_3_layer_Thicc_model((2 * K * M), M, K, i)(input_pass_i)
    model = Model(inputs, x)
    return model
def FDD_softmax_k_times_with_magnitude_rounded(M, K, k):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.abs(inputs)
    input_mod = input_mod * 10.0
    input_mod = tf.keras.layers.Reshape((K * M,))(input_mod)
    # normalize
    mean = tf.expand_dims(tf.reduce_mean(input_mod, axis=1), 1)
    std = tf.expand_dims(tf.math.reduce_std(input_mod, axis=1), 1)
    input_mod = (input_mod - mean) / std
    input_mod = tf.keras.layers.Reshape((input_mod.shape[1],))(input_mod)
    input_mod = input_mod + tf.stop_gradient(tf.round(input_mod * 10) / 10 - input_mod)
    decision_0 = tf.stop_gradient(tf.multiply(tf.zeros((K * M)), input_mod[:, :K * M]))
    input_pass_0 = tf.keras.layers.Concatenate(axis=1)((decision_0, input_mod))
    # dnn_model = DNN_3_layer_model((3*K*M), M, K, 0)
    x = DNN_3_layer_Thicc_model((2 * K * M), M, K, 0)(input_pass_0)
    for i in range(1, k):
        decision_i = x
        input_pass_i = tf.keras.layers.Concatenate(axis=1)((decision_i, input_mod))
        x = x + DNN_3_layer_Thicc_model((2 * K * M), M, K, i)(input_pass_i)
    model = Model(inputs, x)
    return model
def FDD_softmax_k_times(M, K, k):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.keras.layers.Concatenate(axis=2)([tf.math.real(inputs), tf.math.imag(inputs)])
    input_mod = tf.keras.layers.Reshape((2 * K * M,))(input_mod)
    decision_0 = tf.stop_gradient(tf.multiply(tf.zeros((K * M)), input_mod[:, :K * M]))
    input_pass_0 = tf.keras.layers.Concatenate(axis=1)((decision_0, input_mod))
    # dnn_model = DNN_3_layer_model((3*K*M), M, K, 0)
    x = DNN_3_layer_Thicc_model((3 * K * M), M, K, 0)(input_pass_0)
    for i in range(1, k):
        decision_i = x
        input_pass_i = tf.keras.layers.Concatenate(axis=1)((decision_i, input_mod))
        x = x + DNN_3_layer_Thicc_model((3 * K * M), M, K, i)(input_pass_i)
    model = Model(inputs, x)
    return model
def FDD_softmax_k_times_hard_output(M, K, k):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.keras.layers.Concatenate(axis=2)([tf.math.real(inputs), tf.math.imag(inputs)])
    input_mod = tf.keras.layers.Reshape((2 * K * M,))(input_mod)
    decision_0 = tf.stop_gradient(tf.multiply(tf.zeros((K * M)), input_mod[:, :K * M]))
    input_pass_0 = tf.keras.layers.Concatenate(axis=1)((decision_0, input_mod))
    # dnn_model = DNN_3_layer_model((3*K*M), M, K, 0)
    x = DNN_3_layer_Thicc_model((3 * K * M), M, K, 0)(input_pass_0)
    for i in range(1, k):
        decision_i = x + tf.stop_gradient(binary_activation(x, 0.5) - x)
        input_pass_i = tf.keras.layers.Concatenate(axis=1)((decision_i, input_mod))
        output_i = DNN_3_layer_model((3 * K * M), M, K, i)(input_pass_i)
        x = x + output_i
    model = Model(inputs, x)
    return model
def FDD_softmax_k_times_hard_output_with_magnitude(M, K, k):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.abs(inputs)
    input_mod = tf.keras.layers.Reshape((K * M,))(input_mod)
    decision_0 = tf.stop_gradient(tf.multiply(tf.zeros((K * M)), input_mod[:, :K * M]))
    input_pass_0 = tf.keras.layers.Concatenate(axis=1)((decision_0, input_mod))
    # dnn_model = DNN_3_layer_model((3*K*M), M, K, 0)
    x = DNN_3_layer_Thicc_model((2 * K * M), M, K, 0)(input_pass_0)
    for i in range(1, k):
        decision_i = x + tf.stop_gradient(binary_activation(x, 0.5) - x)
        input_pass_i = tf.keras.layers.Concatenate(axis=1)((decision_i, input_mod))
        output_i = DNN_3_layer_model((2 * K * M), M, K, i)(input_pass_i)
        x = x + output_i
    x = x + tf.stop_gradient(binary_activation(x, 0.5) - x)
    model = Model(inputs, x)
    return model
def FDD_softmax_k_times_common_dnn(M, K, k):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.keras.layers.Concatenate(axis=2)([tf.math.real(inputs), tf.math.imag(inputs)])
    input_mod = tf.keras.layers.Reshape((2 * K * M,))(input_mod)
    decision_0 = tf.stop_gradient(tf.multiply(tf.zeros((K * M)), input_mod[:, :K * M]))
    input_pass_0 = tf.keras.layers.Concatenate(axis=1)((decision_0, input_mod))
    dnn_model = DNN_3_layer_model((3 * K * M), M, K, 0)
    x = dnn_model(input_pass_0)
    for i in range(1, k):
        decision_i = x + tf.stop_gradient(binary_activation(x) - x)
        input_pass_i = tf.keras.layers.Concatenate(axis=1)((decision_i, input_mod))
        x = x + dnn_model(input_pass_i)
    model = Model(inputs, x)
    return model
def FDD_ranked_softmax(M, K, k):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.abs(inputs)
    input_mod = tf.keras.layers.Reshape((K * M,))(input_mod)
    decision_0 = tf.stop_gradient(tf.multiply(tf.zeros((K * M)), input_mod[:, :K * M]))
    input_pass_0 = tf.keras.layers.Concatenate(axis=1)((decision_0, input_mod))
    decision_i = DNN_3_layer_model((2 * K * M), M, K, 0)(input_pass_0)
    output = tf.keras.layers.Reshape((M * K, 1))(decision_i)
    for i in range(1, k):
        input_pass_i = tf.keras.layers.Concatenate(axis=1)((decision_i, input_mod))
        input_pass_i = input_pass_i + tf.stop_gradient(binary_activation(input_pass_i, 0.5) - input_pass_i)
        output_i = DNN_3_layer_model((2 * K * M), M, K, i)(input_pass_i)
        decision_i = decision_i + output_i
        output = tf.concat((output, tf.keras.layers.Reshape((decision_i.shape[1], 1))(decision_i)), axis=2)
    model = Model(inputs, output)
    return model
def FDD_harder_softmax_k_times(M, K, k):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.abs(inputs)
    input_mod = tf.keras.layers.Reshape((K * M,))(input_mod)
    decision_0 = tf.stop_gradient(tf.multiply(tf.zeros((K * M)), input_mod[:, :K * M]))
    input_pass_0 = tf.keras.layers.Concatenate(axis=1)((decision_0, input_mod))
    # dnn_model = DNN_3_layer_model((3*K*M), M, K, 0)
    x = DNN_3_layer_model_harder_softmax((2 * K * M), M, K, 0)(input_pass_0)
    for i in range(1, k):
        decision_i = x
        input_pass_i = tf.keras.layers.Concatenate(axis=1)((decision_i, input_mod))
        x = x + DNN_3_layer_model_harder_softmax((2 * K * M), M, K, i)(input_pass_i)
    model = Model(inputs, x)
    return model
def FDD_ranked_softmax_state_change(M, K, k):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.abs(inputs)
    input_mod = tf.keras.layers.Reshape((K * M,))(input_mod)
    decision_0 = tf.stop_gradient(tf.multiply(tf.zeros((K * M)), input_mod[:, :K * M]))
    input_pass_0 = tf.keras.layers.Concatenate(axis=1)((decision_0, input_mod))
    decision_i = DNN_3_layer_model((2 * K * M), M, K, 0)(input_pass_0)
    output = tf.keras.layers.Reshape((M * K, 1))(decision_i)
    input_mod = input_mod * tf.subtract(tf.ones((M * K,)), decision_i)
    for i in range(1, k):
        input_pass_i = tf.keras.layers.Concatenate(axis=1)((decision_i, input_mod))
        input_pass_i = input_pass_i + tf.stop_gradient(binary_activation(input_pass_i, 0.5) - input_pass_i)
        output_i = DNN_3_layer_model((2 * K * M), M, K, i)(input_pass_i)
        decision_i = decision_i + output_i
        output = tf.concat((output, tf.keras.layers.Reshape((decision_i.shape[1], 1))(decision_i)), axis=2)
        input_mod = input_mod * tf.subtract(tf.ones((M * K,)), decision_i)
    model = Model(inputs, output)
    return model
def FDD_ranked_LSTM_softmax(M, K, k):
    lstm_model = tf.keras.layers.LSTMCell(M * K)
    interpreter = DNN_3_layer_model((M * K,), M, K)
    # state_reshaper = tf.keras.layers.Reshape((1, M))
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.abs(inputs)
    input_mod = tf.keras.layers.Reshape((K * M,))(input_mod)
    decision_0 = tf.stop_gradient(tf.multiply(tf.zeros((K * M,)), input_mod[:, :K * M]))
    input_pass_0 = tf.keras.layers.Concatenate(axis=1)((decision_0, input_mod))
    state_0 = tf.stop_gradient(tf.multiply(tf.zeros((K * M,)), input_mod[:, :K * M]))
    state_i, carry_states_i = lstm_model(input_pass_0, [state_0, state_0])
    decision_i = interpreter(state_i)
    # decision_i = tf.keras.layers.Softmax()(state_i)
    output = tf.keras.layers.Reshape((M * K, 1))(decision_i)
    for i in range(1, k):
        input_pass_i = tf.keras.layers.Concatenate(axis=1)((decision_i, input_mod))
        # input_pass_i = input_pass_i + tf.stop_gradient(binary_activation(input_pass_i, 0.5) - input_pass_i)
        state_i, carry_states_i = lstm_model(input_pass_i, carry_states_i)
        output_i = interpreter(state_i)
        # output_i = tf.keras.layers.Softmax()(state_i)
        decision_i = decision_i + output_i
        output = tf.concat((output, tf.keras.layers.Reshape((decision_i.shape[1], 1))(decision_i)), axis=2)
    model = Model(inputs, output)
    return model
def FDD_ranked_softmax_common_DNN(M, K, k):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.abs(inputs)
    input_mod = tf.keras.layers.Reshape((K * M,))(input_mod)
    decision_0 = tf.stop_gradient(tf.multiply(tf.zeros((K * M)), input_mod[:, :K * M]))
    input_pass_0 = tf.keras.layers.Concatenate(axis=1)((decision_0, input_mod))
    network = DNN_3_layer_model((2 * K * M), M, K)
    decision_i = network(input_pass_0)
    output = tf.keras.layers.Reshape((M * K, 1))(decision_i)
    for i in range(1, k):
        input_pass_i = tf.keras.layers.Concatenate(axis=1)((decision_i, input_mod))
        input_pass_i = input_pass_i + tf.stop_gradient(binary_activation(input_pass_i, 0.5) - input_pass_i)
        output_i = network(input_pass_i)
        decision_i = decision_i + output_i
        output = tf.concat((output, tf.keras.layers.Reshape((decision_i.shape[1], 1))(decision_i)), axis=2)
    model = Model(inputs, output)
    return model
def FDD_model_softmax(M, K, B):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    x = tf.keras.layers.Concatenate(axis=2)([tf.math.real(inputs), tf.math.imag(inputs)])
    # create input vector
    reshaper = tf.keras.layers.Reshape((2 * M * K,))
    x = reshaper(x)
    x = Dense(3 * M)(x)
    x = LeakyReLU()(x)
    x = Dense(M)(x)
    x = LeakyReLU()(x)
    x = Dense(M * K)(x)
    # to be removed
    x_list2 = tf.split(x, num_or_size_splits=K, axis=1)
    output = tf.keras.layers.Softmax()(x_list2[0])
    for i in range(1, len(x_list2)):
        output = tf.concat((output, tf.keras.layers.Softmax()(x_list2[i])), axis=1)
    # yep
    # x = tf.sigmoid(x)
    model = Model(inputs, output)
    return model
def FDD_softmax_with_soft_mask(M, K, B, k=3):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    # input_mod = tf.keras.layers.Concatenate(axis=2)([tf.math.real(inputs), tf.math.imag(inputs)])
    input_mod = tf.abs(inputs)
    # create input vector
    reshaper = tf.keras.layers.Reshape((M * K,))
    input_mod = reshaper(input_mod)
    # normalize
    # mean = tf.expand_dims(tf.reduce_mean(input_mod, axis=1), 1)
    # std = tf.expand_dims(tf.math.reduce_std(input_mod, axis=1), 1)
    # x = (input_mod - mean)/std
    # x = tf.keras.layers.Reshape((x.shape[1],))(x)
    # x = x + tf.stop_gradient(tf.round(x * 10)/10 - x)
    # model starts
    x = Dense(M * K)(input_mod)
    x = LeakyReLU()(x)
    x = Dense(M * K)(x)
    x = LeakyReLU()(x)
    x = Dense(M * K)(x)
    # ranking_output = LSTM_Ranking_model(M, K, k)(x)
    ranking_output = DNN_Ranking_model((M * K * 2,), M, K, k, True)(tf.concat((x, input_mod), axis=1))
    x_list2 = tf.split(x, num_or_size_splits=K, axis=1)
    output = tf.keras.layers.Softmax()(x_list2[0])
    for i in range(1, len(x_list2)):
        output = tf.concat((output, tf.keras.layers.Softmax()(x_list2[i])), axis=1)
    # yep
    # x = tf.sigmoid(x)
    output = tf.concat((output, ranking_output), axis=1)
    model = Model(inputs, output)
    print(model.summary())
    return model
def Stochastic_softmax_selectior_and_loss(M, K, N_rf, N=1000):
    def select(y_raw_pred_i, scheduled_output_i, G, lossfn, reg=user_constraint):
        # y_raw_pred: shape(Batchsize, M*K, Nrf)
        # scheduled_output_i: shape(bathsize, M*K)
        mask = np.zeros((y_raw_pred_i.shape[0], N, M*K, N_rf))
        for i in range(0, N_rf):
            sam = tf.random.categorical(y_raw_pred_i[:, :, i], N)
            for batch in range(y_raw_pred_i.shape[0]):
                mask[batch, :, :, i] = tf.one_hot(sam[batch], M*K)
            # for batch in range(y_raw_pred_i.shape[0]):
            #     for n in range(N):
            #         mask[batch, n, sam[batch, n], i] = 1.0
        mask = tf.constant(np.sum(mask, axis=3), dtype=tf.float32)
        scheduled_output_i = tf.expand_dims(scheduled_output_i, axis=1)
        scheduled_output_i = tf.multiply(scheduled_output_i, mask)
        G = tf.tile(tf.expand_dims(G, axis=1), [1,N,1,1])
        G = tf.reshape(G, (G.shape[0] * N, K, M))
        scheduled_output_i = scheduled_output_i + tf.stop_gradient(mask - scheduled_output_i)
        scheduled_output_i = tf.reshape(scheduled_output_i, (scheduled_output_i.shape[0] * N, M*K))
        return lossfn(scheduled_output_i, G) + reg(scheduled_output_i, K, M)
    return select
def FDD_softmax_with_unconstraint_soft_masks(M, K, B, k=3):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    mod_input = tf.abs(inputs)
    # create input vector
    reshaper = tf.keras.layers.Reshape((M * K,))
    mod_input = reshaper(mod_input)
    # normalize
    # mean = tf.expand_dims(tf.reduce_mean(mod_input, axis=1), 1)
    # std = tf.expand_dims(tf.math.reduce_std(mod_input, axis=1), 1)
    # x = (mod_input - mean)/std
    # x = tf.keras.layers.Reshape((x.shape[1],))(x)
    # model starts
    x = Dense(3 * M)(mod_input)
    x = LeakyReLU()(x)
    x = Dense(M)(x)
    x = LeakyReLU()(x)
    x = Dense(M * K)(x)
    # ranking_output = Dense(50)(x)
    # ranking_output = Dense(20)(ranking_output)
    # ranking_output = Dense(k)(ranking_output)
    x_list2 = tf.split(x, num_or_size_splits=K, axis=1)
    output = tf.keras.layers.Softmax()(x_list2[0])
    for i in range(1, len(x_list2)):
        output = tf.concat((output, tf.keras.layers.Softmax()(x_list2[i])), axis=1)
    # yep
    # x = tf.sigmoid(x)
    # ranking_output = tf.keras.layers.Reshape((k*K,))(ranking_output)
    ranking_input = tf.concat((mod_input, x), axis=1)
    ranking_output = Dense(M * K)(ranking_input)
    ranking_output = LeakyReLU()(ranking_output)
    ranking_output = Dense(M)(ranking_output)
    ranking_output = LeakyReLU()(ranking_output)
    ranking_output = Dense(K)(ranking_output)
    ranking_output = sigmoid(ranking_output)
    # ranking_output = tf.tanh(tf.keras.layers.ReLU()(ranking_output))
    output = tf.concat((output, ranking_output), axis=1)
    model = Model(inputs, output)
    return model
def FDD_k_times_with_sigmoid_and_penalty(M, K, k=3):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.abs(inputs)
    input_mod = tf.keras.layers.Reshape((K * M,))(input_mod)
    decision_0 = tf.stop_gradient(tf.multiply(tf.zeros((K * M)), input_mod[:, :K * M]))
    input_pass_0 = tf.keras.layers.Concatenate(axis=1)((decision_0, input_mod))
    dnn_model = DNN_3_layer_model_sigmoid((2 * K * M), M, K, 0)
    x = dnn_model(input_pass_0)
    for i in range(1, k):
        decision_i = x
        input_pass_i = tf.keras.layers.Concatenate(axis=1)((decision_i, input_mod))
        x = dnn_model(input_pass_i)
    model = Model(inputs, x)
    return model
def dnn_per_link(input_shape, N_rf, i=0):
    inputs = Input(shape=input_shape, name="DNN_input_insideDNN{}".format(i))
    x = Dense(64, name="Dense1_inside_DNN{}".format(i))(inputs)
    x = tf.keras.layers.BatchNormalization(name="batchnorm_inside_DNN{}".format(i))(x)
    x = sigmoid(x)
    x = Dense(64, name="Dense2_inside_DNN{}".format(i))(x)
    x = tf.keras.layers.BatchNormalization(name="batchnorm_inside_DNN_2{}".format(i))(x)
    x = sigmoid(x)
    # x = Dense(32, name="Dense3_inside_DNN{}".format(i))(x)
    # x = tf.keras.layers.BatchNormalization(name="batchnorm_inside_DNN_4{}".format(i))(x)
    # x = sigmoid(x)
    x = Dense(N_rf, name="Dense4_inside_DNN{}".format(i))(x)
    # x = sigmoid(x)
    model = Model(inputs, x, name="DNN_within_model{}".format(i))
    return model
def dnn_per_link_mutex(input_shape, N_rf):
    inputs = Input(shape=input_shape)
    x = Dense(512)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = sigmoid(x)
    # x = Dense(256)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = sigmoid(x)
    x = Dense(N_rf)(x)
    mutex = tf.eye(N_rf) - tf.ones((N_rf, N_rf))
    mutex = tf.expand_dims(mutex, axis=0)
    mutex = tf.expand_dims(mutex, axis=0)
    x = tf.expand_dims(x, axis=3)
    x = tf.multiply(x, sigmoid(tf.matmul(mutex, x)))[:, :, :, 0]
    # x = sigmoid(x)
    model = Model(inputs, x)
    return model
def dnn_sequential(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(128)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = sigmoid(x)
    x = Dense(1)(x)
    # x = sigmoid(x)
    model = Model(inputs, x)
    return model
def FDD_per_link_archetecture_more_granular(M, K, k=2, N_rf=3, output_all=False):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.square(tf.abs(inputs))
    input_mod = tf.keras.layers.BatchNormalization()(input_mod)
    input_modder = Interference_Input_modification(K, M, N_rf, k)
    dnns = dnn_per_link((M * K, 4 + M * K), N_rf)
    # compute interference from k,i
    output_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0 * N_rf / M / K)
    input_i = input_modder(output_0, input_mod, k - 1.0)
    raw_out_put_i = dnns(input_i)
    raw_out_put_i = tf.keras.layers.Softmax(axis=1)(raw_out_put_i) # (None, K*M, Nrf)
    # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
    out_put_i = tf.reduce_sum(raw_out_put_i, axis=2) # (None, K*M)
    output = [tf.expand_dims(out_put_i, axis=1), tf.expand_dims(raw_out_put_i, axis=1)]
    # begin the second - kth iteration
    for times in range(1, k):
        out_put_i = tf.keras.layers.Reshape((K, M))(out_put_i)
        input_i = input_modder(out_put_i, input_mod, k - times - 1.0)
        raw_out_put_i = dnns(input_i)
        raw_out_put_i = tf.keras.layers.Softmax(axis=1)(raw_out_put_i)
        # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
        out_put_i = tf.reduce_sum(raw_out_put_i, axis=2)
        output[0] = tf.concat([output[0], tf.expand_dims(out_put_i, axis=1)], axis=1)
        output[1] = tf.concat([output[1], tf.expand_dims(raw_out_put_i, axis=1)], axis=1)
    model = Model(inputs, output)
    return model
def FDD_per_link_archetecture_more_G_sigmoid(M, K, k=2, N_rf=3, normalization=True, avg_max=None):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.abs(inputs)
    # input_mod = tf.keras.layers.BatchNormalization()(input_mod)
    if normalization:
        input_mod = tf.divide(input_mod, avg_max)
    input_mod = tf.square(input_mod)
    # input_modder = Per_link_Input_modification_most_G(K, M, N_rf, k)
    input_modder = Per_link_Input_modification_most_G_raw_self_sigmoid(K, M, N_rf, k)
    # input_modder = Per_link_Input_modification_most_G(K, M, N_rf, k)
    # input_modder = Per_link_Input_modification_most_G_col_lessX(K, M, N_rf, k)
    # sm = Argmax_SPIGOT_layer()
    # sm = Argmax_STE_layer()
    # sm = Sparsemax(axis=1)
    # input_modder = Per_link_Input_modification_learnable_G(K, M, N_rf, k)
    dnns = dnn_per_link((M * K ,13 + 1), 1)
    # dnns = dnn_per_link((M * K, 13 + 3*K), N_rf)
    # compute interference from k,i
    # output_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0 * N_rf / M / K)
    raw_out_put_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0)
    raw_out_put_0 = tf.tile(tf.expand_dims(raw_out_put_0, axis=3), (1, 1, 1, 1))
    raw_out_put_0 = tf.keras.layers.Reshape((K*M, 1))(raw_out_put_0)
    input_i = input_modder(raw_out_put_0, input_mod, k - 1.0)
    # input_i = input_modder(output_0, input_mod, k - 1.0)
    raw_out_put_i = dnns(input_i)
    raw_out_put_i = sigmoid(raw_out_put_i) # (None, K*M, Nrf)
    # raw_out_put_i = sigmoid((raw_out_put_i - 0.4) * 20.0)
    # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
    out_put_i = tf.reduce_sum(raw_out_put_i, axis=2) # (None, K*M)
    output = [tf.expand_dims(out_put_i, axis=1), tf.expand_dims(raw_out_put_i, axis=1)]
    # begin the second - kth iteration
    for times in range(1, k):
        out_put_i = tf.keras.layers.Reshape((K, M))(out_put_i)
        # input_mod_temp = tf.multiply(out_put_i, input_mod) + input_mod
        input_i = input_modder(raw_out_put_i, input_mod, k - times - 1.0)
        # input_i = input_modder(out_put_i, input_mod, k - times - 1.0)
        raw_out_put_i = dnns(input_i)
        raw_out_put_i = sigmoid(raw_out_put_i)
        out_put_i = tf.reduce_sum(raw_out_put_i, axis=2)
        # raw_out_put_i = sigmoid((raw_out_put_i - 0.4) * 20.0)
        # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
        output[0] = tf.concat([output[0], tf.expand_dims(out_put_i, axis=1)], axis=1)
        output[1] = tf.concat([output[1], tf.expand_dims(raw_out_put_i, axis=1)], axis=1)
    model = Model(inputs, output)
    return model
def FDD_per_link_archetecture_more_G(M, K, k=2, N_rf=3, normalization=True, avg_max=None):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.abs(inputs)
    # input_mod = tf.keras.layers.BatchNormalization()(input_mod)
    if normalization:
        input_mod = tf.divide(input_mod, avg_max)
    input_mod = tf.square(input_mod)

    # input_modder = Per_link_Input_modification_most_G(K, M, N_rf, k)
    input_modder = Per_link_Input_modification_most_G_raw_self(K, M, N_rf, k)
    # input_modder = Per_link_Input_modification_most_G(K, M, N_rf, k)
    # input_modder = Per_link_Input_modification_most_G_col_lessX(K, M, N_rf, k)
    sm = tf.keras.layers.Softmax(axis=1)
    # sm = Argmax_SPIGOT_layer()
    # sm = Argmax_STE_layer()
    # sm = Sparsemax(axis=1)
    # input_modder = Per_link_Input_modification_learnable_G(K, M, N_rf, k)
    dnns = dnn_per_link((M * K ,16 + N_rf), N_rf)
    # dnns = dnn_per_link((M * K, 13 + 3*K), N_rf)
    # compute interference from k,i
    # output_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0 * N_rf / M / K)
    raw_out_put_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0)
    raw_out_put_0 = tf.tile(tf.expand_dims(raw_out_put_0, axis=3), (1, 1, 1, N_rf))
    raw_out_put_0 = tf.keras.layers.Reshape((K*M, N_rf))(raw_out_put_0)
    input_i = input_modder(raw_out_put_0, input_mod, k - 1.0)
    # input_i = input_modder(output_0, input_mod, k - 1.0)
    raw_out_put_i = dnns(input_i)
    # raw_out_put_i = sigmoid((raw_out_put_i - 0.4) * 20.0)
    # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
    out_put_i = tf.reduce_sum(sm(raw_out_put_i), axis=2) # (None, K*M)
    output = [tf.expand_dims(out_put_i, axis=1), tf.expand_dims(raw_out_put_i, axis=1)]
    # begin the second - kth iteration
    for times in range(1, k):
        out_put_i = tf.keras.layers.Reshape((K, M))(out_put_i)
        # input_mod_temp = tf.multiply(out_put_i, input_mod) + input_mod
        input_i = input_modder(raw_out_put_i, input_mod, k - times - 1.0)
        # input_i = input_modder(out_put_i, input_mod, k - times - 1.0)
        raw_out_put_i = dnns(input_i)
        out_put_i = tf.reduce_sum(sm(raw_out_put_i), axis=2)
        # raw_out_put_i = sigmoid((raw_out_put_i - 0.4) * 20.0)
        # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
        output[0] = tf.concat([output[0], tf.expand_dims(out_put_i, axis=1)], axis=1)
        output[1] = tf.concat([output[1], tf.expand_dims(raw_out_put_i, axis=1)], axis=1)
    model = Model(inputs, output, name="scheduler")
    return model
def FDD_ensemble_model(M, K, k=2, N_rf=3, normalization=True, avg_max=None):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    model_1 = FDD_agent_more_G(M, K, k, N_rf, normalization, avg_max, i=0)
    model_2 = FDD_agent_more_G(M, K, k, N_rf, normalization, avg_max, i=1)
    out_1 = model_1(inputs)
    out_2 = model_2(inputs)
    model = Model(inputs, [out_1[0], out_1[1], out_2[0], out_2[1]], name="ensemble")
    return model

def train_with_pretrai():
    model1 = tf.keras.layers.Dense(out_size1, name="dense1")
    model2 = tf.keras.layers.Dense(out_size2, name="dense2")

    # train model 2
    # ...
    # ...
    model2.save("name.h5")

    # now to build a new model with model1 and model2
    # load model:
    model2_trained = tf.keras.models.load_model("name.h5")
    model2_trained
    # build new model:
    newmodel_input = Input(shape=(input_size, ))
    out = model2(model1(newmodel_input))
    end_to_end_model = Model(newmodel_input, out, name="endtoendModel")
    end_to_end_model.get_layer("dense2").trainable = False
    # if trainable is set to False Dense2 would not be updated

def FDD_agent_more_G_with_weights(M, K, k=2, N_rf=3, normalization=True, avg_max=None, i=0):
    def self_agent_dnn(input_shape, i=i):
        inputs = Input(shape=input_shape, name="DNN_input_insideDNN{}".format(i))
        x = Dense(64, name="Dense1_inside_DNN{}".format(i))(inputs)
        x = tf.keras.layers.BatchNormalization(name="batchnorm_inside_DNN{}".format(i))(x)
        # x = tf.keras.layers.Dropout(0.1)(x)
        x = sigmoid(x)
        # x = tf.math.log(1+tf.exp(x))
        x = Dense(64, name="Dense2_inside_DNN{}".format(i))(x)
        x = tf.keras.layers.BatchNormalization(name="batchnorm_inside_DNN_2{}".format(i))(x)
        # x = tf.keras.layers.Dropout(0.1)(x)
        x = sigmoid(x)
        # x = tf.math.log(1+tf.exp(x))
        x = Dense(N_rf, name="Dense4_inside_DNN{}".format(i))(x)
        model = Model(inputs, x, name="DNN_within_model{}".format(i))
        return model
    inputs = Input(shape=(K, M + 1), dtype=tf.complex64)
    input_mod = inputs[:, :, :M]
    weights = tf.abs(inputs[:, :, M:])
    weights = weights/tf.reduce_max(weights, axis=1, keepdims=True)
    input_mod = tf.abs(input_mod)
    if normalization:
        input_mod = tf.divide(input_mod, avg_max)
    input_mod = tf.square(input_mod)
    input_modder = Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights_different_weights(K, M, N_rf, k)
    sm = tf.keras.layers.Softmax(axis=1)
    # sm = sigmoid
    dnns = self_agent_dnn((M * K ,21 + N_rf))
    raw_out_put_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0)
    raw_out_put_0 = tf.tile(tf.expand_dims(raw_out_put_0, axis=3), (1, 1, 1, N_rf))
    raw_out_put_0 = tf.keras.layers.Reshape((K*M, N_rf))(raw_out_put_0)
    input_i = input_modder(raw_out_put_0, input_mod, weights)
    # input_i = input_modder(output_0, input_mod, k - 1.0)
    raw_out_put_i = dnns(input_i)

    out_put_i = tf.reduce_sum(sm(raw_out_put_i), axis=2) # (None, K*M)
    # out_put_i = tf.reduce_sum(sigmoid(raw_out_put_i), axis=2)  # (None, K*M)

    output = [tf.expand_dims(out_put_i, axis=1), tf.expand_dims(raw_out_put_i, axis=1)]
    # begin the second - kth iteration
    for times in range(1, k):
        out_put_i = tf.keras.layers.Reshape((K, M))(out_put_i)
        # input_mod_temp = tf.multiply(out_put_i, input_mod) + input_mod
        input_i = input_modder(raw_out_put_i, input_mod, weights)
        # input_i = input_modder(out_put_i, input_mod, k - times - 1.0)
        raw_out_put_i = dnns(input_i)
        # if times == k-1:
        out_put_i = tf.reduce_sum(sm(raw_out_put_i), axis=2)
        # else:
        #     out_put_i = tf.reduce_sum(sigmoid(raw_out_put_i), axis=2)
        # raw_out_put_i = sigmoid((raw_out_put_i - 0.4) * 20.0)
        # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
        output[0] = tf.concat([output[0], tf.expand_dims(out_put_i, axis=1)], axis=1)
        output[1] = tf.concat([output[1], tf.expand_dims(raw_out_put_i, axis=1)], axis=1)
    model = Model(inputs, output, name="scheduler{}".format(i))
    return model
def FDD_agent_more_G(M, K, k=2, N_rf=3, normalization=True, avg_max=None, i=0):
    def self_agent_dnn(input_shape, i=i):
        inputs = Input(shape=input_shape, name="DNN_input_insideDNN{}".format(i))
        x = Dense(64, name="Dense1_inside_DNN{}".format(i))(inputs)
        x = tf.keras.layers.BatchNormalization(name="batchnorm_inside_DNN{}".format(i))(x)
        x = sigmoid(x)
        # x = tf.math.log(1+tf.exp(x))
        x = Dense(64, name="Dense2_inside_DNN{}".format(i))(x)
        x = tf.keras.layers.BatchNormalization(name="batchnorm_inside_DNN_2{}".format(i))(x)
        x = sigmoid(x)
        # x = tf.math.log(1+tf.exp(x))
        x = Dense(N_rf, name="Dense4_inside_DNN{}".format(i))(x)
        model = Model(inputs, x, name="DNN_within_model{}".format(i))
        return model
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.abs(inputs)
    if normalization:
        input_mod = tf.divide(input_mod, avg_max)
    input_mod = tf.square(input_mod)
    input_modder = Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum(K, M, N_rf, k)
    sm = tf.keras.layers.Softmax(axis=1)
    # sm = sigmoid
    dnns = self_agent_dnn((M * K ,17 + N_rf))
    raw_out_put_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0)
    raw_out_put_0 = tf.tile(tf.expand_dims(raw_out_put_0, axis=3), (1, 1, 1, N_rf))
    raw_out_put_0 = tf.keras.layers.Reshape((K*M, N_rf))(raw_out_put_0)
    input_i = input_modder(raw_out_put_0, input_mod, k - 1.0)
    # input_i = input_modder(output_0, input_mod, k - 1.0)
    raw_out_put_i = dnns(input_i)

    out_put_i = tf.reduce_sum(sm(raw_out_put_i), axis=2) # (None, K*M)
    # out_put_i = tf.reduce_sum(sigmoid(raw_out_put_i), axis=2)  # (None, K*M)

    output = [tf.expand_dims(out_put_i, axis=1), tf.expand_dims(raw_out_put_i, axis=1)]
    # begin the second - kth iteration
    for times in range(1, k):
        out_put_i = tf.keras.layers.Reshape((K, M))(out_put_i)
        # input_mod_temp = tf.multiply(out_put_i, input_mod) + input_mod
        input_i = input_modder(raw_out_put_i, input_mod, k - times - 1.0)
        # input_i = input_modder(out_put_i, input_mod, k - times - 1.0)
        raw_out_put_i = dnns(input_i)
        # if times == k-1:
        out_put_i = tf.reduce_sum(sm(raw_out_put_i), axis=2)
        # else:
        #     out_put_i = tf.reduce_sum(sigmoid(raw_out_put_i), axis=2)
        # raw_out_put_i = sigmoid((raw_out_put_i - 0.4) * 20.0)
        # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
        output[0] = tf.concat([output[0], tf.expand_dims(out_put_i, axis=1)], axis=1)
        output[1] = tf.concat([output[1], tf.expand_dims(raw_out_put_i, axis=1)], axis=1)
    model = Model(inputs, output, name="scheduler{}".format(i))
    return model
def FDD_agent_more_G_with_moderator(M, K, k=2, N_rf=3, normalization=True, avg_max=None, i=0):
    def self_agent_dnn(input_shape, i=i):
        inputs = Input(shape=input_shape, name="DNN_input_insideDNN{}".format(i))
        x = Dense(64, name="Dense1_inside_DNN{}".format(i))(inputs)
        x = tf.keras.layers.BatchNormalization(name="batchnorm_inside_DNN{}".format(i))(x)
        x = sigmoid(x)
        x = Dense(64, name="Dense2_inside_DNN{}".format(i))(x)
        x = tf.keras.layers.BatchNormalization(name="batchnorm_inside_DNN_2{}".format(i))(x)
        x = sigmoid(x)
        x = Dense(N_rf, name="Dense4_inside_DNN{}".format(i))(x)
        model = Model(inputs, x, name="DNN_within_model{}".format(i))
        return model
    def moderator():
        inputs = Input(shape=[N_rf, M*K])
        x = Dense(64)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = sigmoid(x)
        x = Dense(K*M)(x)
        model = Model(inputs, x, name="moderator")
        return model
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.abs(inputs)
    if normalization:
        input_mod = tf.divide(input_mod, avg_max)
    input_mod = tf.square(input_mod)
    input_modder = Per_link_Input_modification_most_G_raw_self_more_interference(K, M, N_rf, k)
    sm = tf.keras.layers.Softmax(axis=1)
    # sm = sigmoid
    dnns = self_agent_dnn((M * K ,17 + N_rf))
    moderator_dnn = moderator()
    raw_out_put_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0)
    raw_out_put_0 = tf.tile(tf.expand_dims(raw_out_put_0, axis=3), (1, 1, 1, N_rf))
    raw_out_put_0 = tf.keras.layers.Reshape((K*M, N_rf))(raw_out_put_0)
    input_i = input_modder(raw_out_put_0, input_mod, k - 1.0)
    # input_i = input_modder(output_0, input_mod, k - 1.0)
    raw_out_put_i = dnns(input_i)
    raw_out_put_sig = moderator_dnn(tf.transpose(raw_out_put_i, [0, 2, 1]))
    raw_out_put_sig = sigmoid(tf.transpose(raw_out_put_sig, [0, 2, 1]))
    raw_out_put_i = tf.multiply(raw_out_put_i, raw_out_put_sig)
    out_put_i = tf.reduce_sum(sm(raw_out_put_i), axis=2) # (None, K*M)
    # out_put_i = tf.reduce_sum(sigmoid(raw_out_put_i), axis=2)  # (None, K*M)

    output = [tf.expand_dims(out_put_i, axis=1), tf.expand_dims(raw_out_put_i, axis=1)]
    # begin the second - kth iteration
    for times in range(1, k):
        out_put_i = tf.keras.layers.Reshape((K, M))(out_put_i)
        # input_mod_temp = tf.multiply(out_put_i, input_mod) + input_mod
        input_i = input_modder(raw_out_put_i, input_mod, k - times - 1.0)
        # input_i = input_modder(out_put_i, input_mod, k - times - 1.0)
        raw_out_put_i = dnns(input_i)
        raw_out_put_sig = moderator_dnn(tf.transpose(raw_out_put_i, [0, 2, 1]))
        raw_out_put_sig = sigmoid(tf.transpose(raw_out_put_sig, [0, 2, 1]))
        raw_out_put_i = tf.multiply(raw_out_put_i, raw_out_put_sig)

        # if times == k-1:
        out_put_i = tf.reduce_sum(sm(raw_out_put_i), axis=2)
        # else:
        #     out_put_i = tf.reduce_sum(sigmoid(raw_out_put_i), axis=2)
        # raw_out_put_i = sigmoid((raw_out_put_i - 0.4) * 20.0)
        # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
        output[0] = tf.concat([output[0], tf.expand_dims(out_put_i, axis=1)], axis=1)
        output[1] = tf.concat([output[1], tf.expand_dims(raw_out_put_i, axis=1)], axis=1)
    model = Model(inputs, output, name="scheduler{}".format(i))
    return model
def FDD_per_link_archetecture_more_G_original(M, K, k=2, N_rf=3, normalization=True, avg_max=None):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.abs(inputs)
    # input_mod = tf.keras.layers.BatchNormalization()(input_mod)
    if normalization:
        input_mod = tf.divide(input_mod, avg_max)
    input_mod = tf.square(input_mod)

    # input_modder = Per_link_Input_modification_most_G(K, M, N_rf, k)
    input_modder = Per_link_Input_modification_most_G_original(K, M, N_rf, k)
    sm = tf.keras.layers.Softmax(axis=1)
    # sm = Argmax_SPIGOT_layer()
    # sm = Argmax_STE_layer()
    # sm = Sparsemax(axis=1)
    # input_modder = Per_link_Input_modification_learnable_G(K, M, N_rf, k)
    dnns = dnn_per_link((M * K ,5 + M*K), N_rf)
    # dnns = dnn_per_link((M * K, 13 + 3*K), N_rf)
    # compute interference from k,i
    # output_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0 * N_rf / M / K)
    raw_out_put_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0)
    raw_out_put_0 = tf.tile(tf.expand_dims(raw_out_put_0, axis=3), (1, 1, 1, N_rf))
    raw_out_put_0 = tf.keras.layers.Reshape((K*M, N_rf))(raw_out_put_0)
    input_i = input_modder(raw_out_put_0, input_mod, k - 1.0)
    # input_i = input_modder(output_0, input_mod, k - 1.0)
    raw_out_put_i = dnns(input_i)
    raw_out_put_i = sm(raw_out_put_i) # (None, K*M, Nrf)
    # raw_out_put_i = sigmoid((raw_out_put_i - 0.4) * 20.0)
    # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
    out_put_i = tf.reduce_sum(raw_out_put_i, axis=2) # (None, K*M)
    output = [tf.expand_dims(out_put_i, axis=1), tf.expand_dims(raw_out_put_i, axis=1)]
    # begin the second - kth iteration
    for times in range(1, k):
        out_put_i = tf.keras.layers.Reshape((K, M))(out_put_i)
        # input_mod_temp = tf.multiply(out_put_i, input_mod) + input_mod
        input_i = input_modder(raw_out_put_i, input_mod, k - times - 1.0)
        # input_i = input_modder(out_put_i, input_mod, k - times - 1.0)
        raw_out_put_i = dnns(input_i)
        raw_out_put_i = sm(raw_out_put_i)
        out_put_i = tf.reduce_sum(raw_out_put_i, axis=2)
        # raw_out_put_i = sigmoid((raw_out_put_i - 0.4) * 20.0)
        # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
        output[0] = tf.concat([output[0], tf.expand_dims(out_put_i, axis=1)], axis=1)
        output[1] = tf.concat([output[1], tf.expand_dims(raw_out_put_i, axis=1)], axis=1)
    model = Model(inputs, output, name="scheduler")
    return model
def FDD_per_link_archetecture_more_G_return_input_mod(M, K, k=2, N_rf=3, normalization=True, avg_max=None):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.abs(inputs)
    inputs_original = Input((K, M))
    inputs_original_mod = tf.abs(inputs_original)
    # input_mod = tf.keras.layers.BatchNormalization()(input_mod)
    if normalization:
        input_mod = tf.divide(input_mod, avg_max)
    input_mod = tf.square(input_mod)
    inputs_original_mod = tf.square(inputs_original_mod)
    # input_modder = Per_link_Input_modification_most_G(K, M, N_rf, k)
    input_modder = Per_link_Input_modification_most_G_raw_self(K, M, N_rf, k)
    # input_modder = Per_link_Input_modification_most_G(K, M, N_rf, k)
    # input_modder = Per_link_Input_modification_most_G_col_lessX(K, M, N_rf, k)

    sm = tf.keras.layers.Softmax(axis=1)
    # sm = Argmax_SPIGOT_layer()
    # sm = Argmax_STE_layer()
    # sm = Sparsemax(axis=1)
    # input_modder = Per_link_Input_modification_learnable_G(K, M, N_rf, k)
    dnns = dnn_per_link((M * K ,13 + N_rf), N_rf)
    # dnns = dnn_per_link((M * K, 13 + 3*K), N_rf)
    # compute interference from k,i
    # output_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0 * N_rf / M / K)
    raw_out_put_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0)
    raw_out_put_0 = tf.tile(tf.expand_dims(raw_out_put_0, axis=3), (1, 1, 1, N_rf))
    raw_out_put_0 = tf.keras.layers.Reshape((K*M, N_rf))(raw_out_put_0)
    input_0 = input_modder(raw_out_put_0, input_mod, k - 1.0)
    input_r_0 = input_modder(raw_out_put_0, inputs_original_mod, k-1.0)
    # input_i = input_modder(output_0, input_mod, k - 1.0)
    raw_out_put_i = dnns(input_0)
    raw_out_put_i = sm(raw_out_put_i) # (None, K*M, Nrf)
    # raw_out_put_i = sigmoid((raw_out_put_i - 0.4) * 20.0)
    # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
    out_put_i = tf.reduce_sum(raw_out_put_i, axis=2) # (None, K*M)
    output = [tf.expand_dims(out_put_i, axis=1), tf.expand_dims(raw_out_put_i, axis=1)]
    # begin the second - kth iteration
    for times in range(1, k):
        out_put_i = tf.keras.layers.Reshape((K, M))(out_put_i)
        # input_mod_temp = tf.multiply(out_put_i, input_mod) + input_mod
        input_i = input_modder(raw_out_put_i, input_mod, k - times - 1.0)
        # input_i = input_modder(out_put_i, input_mod, k - times - 1.0)
        raw_out_put_i = dnns(input_i)
        raw_out_put_i = sm(raw_out_put_i)
        out_put_i = tf.reduce_sum(raw_out_put_i, axis=2)
        # raw_out_put_i = sigmoid((raw_out_put_i - 0.4) * 20.0)
        # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
        output[0] = tf.concat([output[0], tf.expand_dims(out_put_i, axis=1)], axis=1)
        output[1] = tf.concat([output[1], tf.expand_dims(raw_out_put_i, axis=1)], axis=1)
    output.append(input_0)
    output.append(input_r_0)
    model = Model([inputs, inputs_original], output, name="scheduler")
    return model
def FDD_per_link_archetecture_more_G_logit(M, K, k=2, N_rf=3, normalization=True, avg_max=None):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.abs(inputs)
    # input_mod = tf.keras.layers.BatchNormalization()(input_mod)
    if normalization:
        input_mod = tf.divide(input_mod, avg_max)
    input_mod = tf.square(input_mod)
    # input_modder = Per_link_Input_modification_most_G(K, M, N_rf, k)
    input_modder = Per_link_Input_modification_most_G_raw_self(K, M, N_rf, k)
    # input_modder = Per_link_Input_modification_most_G(K, M, N_rf, k)
    # input_modder = Per_link_Input_modification_most_G_col_lessX(K, M, N_rf, k)

    sm = tf.keras.layers.Softmax(axis=1)
    # sm = Argmax_SPIGOT_layer()
    # sm = Argmax_STE_layer()
    # sm = Sparsemax(axis=1)
    # input_modder = Per_link_Input_modification_learnable_G(K, M, N_rf, k)
    dnns = dnn_per_link((M * K ,13 + N_rf), N_rf)
    # dnns = dnn_per_link((M * K, 13 + 3*K), N_rf)
    # compute interference from k,i
    # output_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0 * N_rf / M / K)
    raw_out_put_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0)
    raw_out_put_0 = tf.tile(tf.expand_dims(raw_out_put_0, axis=3), (1, 1, 1, N_rf))
    raw_out_put_0 = tf.keras.layers.Reshape((K*M, N_rf))(raw_out_put_0)
    input_i = input_modder(raw_out_put_0, input_mod, k - 1.0)
    # input_i = input_modder(output_0, input_mod, k - 1.0)
    raw_out_put_i = dnns(input_i)
    # raw_out_put_i = sigmoid((raw_out_put_i - 0.4) * 20.0)
    # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
    out_put_i = tf.reduce_sum(sm(raw_out_put_i), axis=2) # (None, K*M)
    output = [tf.expand_dims(out_put_i, axis=1), tf.expand_dims(raw_out_put_i, axis=1)]
    # begin the second - kth iteration
    for times in range(1, k):
        out_put_i = tf.keras.layers.Reshape((K, M))(out_put_i)
        # input_mod_temp = tf.multiply(out_put_i, input_mod) + input_mod
        input_i = input_modder(raw_out_put_i, input_mod, k - times - 1.0)
        # input_i = input_modder(out_put_i, input_mod, k - times - 1.0)
        raw_out_put_i = dnns(input_i)
        out_put_i = tf.reduce_sum(sm(raw_out_put_i), axis=2)
        # raw_out_put_i = sigmoid((raw_out_put_i - 0.4) * 20.0)
        # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
        output[0] = tf.concat([output[0], tf.expand_dims(out_put_i, axis=1)], axis=1)
        output[1] = tf.concat([output[1], tf.expand_dims(raw_out_put_i, axis=1)], axis=1)
    model = Model(inputs, output)
    return model
def FDD_per_link_2Fold(M, K, k=2, N_rf=3, output_all=False):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.square(tf.abs(inputs))

    norm = tf.reduce_max(tf.keras.layers.Reshape((K*M, ))(input_mod), axis=1, keepdims=True)
    input_mod = tf.divide(input_mod, tf.expand_dims(norm, axis=1))
    input_modder = Per_link_Input_modification_most_G_no_x(K, M, N_rf, k)
    layer2Modder = Per_link_Input_modification_most_G_col(K, M, N_rf, k)
    # input_modder = Per_link_Input_modification_most_G(K, M, N_rf, k)
    sm = Sparsemax(axis=1)
    # sm = Argmax_SPIGOT_layer()
    # input_modder = Per_link_Input_modification_learnable_G(K, M, N_rf, k)
    dnn1 = dnn_per_link((M * K ,9+K), 1, 0)
    dnn2 = dnn_per_link((M * K ,10+ M*K + K), N_rf, 1)
    # compute interference from k,i
    output_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0 * N_rf / M / K)
    # raw_out_put_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0 / M / K)
    # raw_out_put_0 = tf.tile(tf.expand_dims(raw_out_put_0, axis=3), (1, 1, 1, N_rf))
    # raw_out_put_0 = tf.keras.layers.Reshape((K*M, N_rf))(raw_out_put_0)
    input_i = input_modder(output_0, input_mod, k - 1.0)
    raw_out_put_i = dnn1(input_i)
    raw_out_put_i = tf.keras.layers.Softmax(axis=1)(raw_out_put_i) # (None, K*M, Nrf)
    out_put_i = tf.reduce_sum(raw_out_put_i, axis=2)
    input_i = layer2Modder(tf.keras.layers.Reshape((K, M))(out_put_i), input_mod, 0.0)

    raw_out_put_i = dnn2(input_i)
    raw_out_put_i = sm(raw_out_put_i)  # (None, K*M, Nrf)
    out_put_i = tf.reduce_sum(raw_out_put_i, axis=2)  # (None, K*M)
    output = [tf.expand_dims(out_put_i, axis=1), tf.expand_dims(raw_out_put_i, axis=1)] #(None, 1, K*M)
    model = Model(inputs, output)
    return model
def FDD_per_link_archetecture_more_G_with_argmax(M, K, k=2, N_rf=3, output_all=False):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.square(tf.abs(inputs))
    argmax_with_grad = Sparsemax(axis=1)
    norm = tf.reduce_max(tf.keras.layers.Reshape((K*M, ))(input_mod), axis=1, keepdims=True)
    input_mod = tf.divide(input_mod, tf.expand_dims(norm, axis=1))
    # input_mod = tf.keras.layers.BatchNormalization()(input_mod)
    input_modder = Per_link_Input_modification_most_G_col(K, M, N_rf, k)
    # input_modder = Per_link_Input_modification_most_G(K, M, N_rf, k)
    # input_modder = Per_link_Input_modification_learnable_G(K, M, N_rf, k)
    dnns = dnn_per_link((M * K ,10+ M*K + K), N_rf)
    # compute interference from k,i
    output_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0 * N_rf / M / K)
    # raw_out_put_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0 / M / K)
    # raw_out_put_0 = tf.tile(tf.expand_dims(raw_out_put_0, axis=3), (1, 1, 1, N_rf))
    # raw_out_put_0 = tf.keras.layers.Reshape((K*M, N_rf))(raw_out_put_0)
    input_i = input_modder(output_0, input_mod, k - 1.0)
    # input_i = input_modder(output_0, input_mod, k - 1.0)
    raw_out_put_i = dnns(input_i)
    raw_out_put_i = argmax_with_grad(raw_out_put_i) # (None, K*M, Nrf)
    # raw_out_put_i = sigmoid((raw_out_put_i - 0.4) * 20.0)
    # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
    out_put_i = tf.reduce_sum(raw_out_put_i, axis=2) # (None, K*M)
    output = [tf.expand_dims(out_put_i, axis=1), tf.expand_dims(raw_out_put_i, axis=1)]
    # begin the second - kth iteration
    for times in range(1, k):
        out_put_i = tf.keras.layers.Reshape((K, M))(out_put_i)
        # input_mod_temp = tf.multiply(out_put_i, input_mod) + input_mod
        input_i = input_modder(out_put_i, input_mod, k - times - 1.0)
        # input_i = input_modder(out_put_i, input_mod, k - times - 1.0)
        raw_out_put_i = dnns(input_i)
        raw_out_put_i = argmax_with_grad(raw_out_put_i)
        # if times < k-1:
        #     raw_out_put_i = tf.keras.layers.Softmax(axis=1)(raw_out_put_i)
        # else:
        #     raw_out_put_i = argmax_with_grad(raw_out_put_i)
        # raw_out_put_i = sigmoid((raw_out_put_i - 0.4) * 20.0)
        # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
        out_put_i = tf.reduce_sum(raw_out_put_i, axis=2)
        output[0] = tf.concat([output[0], tf.expand_dims(out_put_i, axis=1)], axis=1)
        output[1] = tf.concat([output[1], tf.expand_dims(raw_out_put_i, axis=1)], axis=1)
    model = Model(inputs, output)
    return model
def FDD_per_link_archetecture_more_G_diff_SM(M, K, k=2, N_rf=3, output_all=False):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.square(tf.abs(inputs))
    norm = tf.reduce_max(tf.keras.layers.Reshape((K*M, ))(input_mod), axis=1, keepdims=True)
    input_mod = tf.divide(input_mod, tf.expand_dims(norm, axis=1))
    # input_mod = tf.keras.layers.BatchNormalization()(input_mod)
    # input_modder = Per_link_Input_modification_most_G_raw_self(K, M, N_rf, k)
    input_modder = Per_link_Input_modification_most_G(K, M, N_rf, k)
    sm = tf.keras.layers.Softmax(axis=1)
    # input_modder = Per_link_Input_modification_learnable_G(K, M, N_rf, k)
    dnns = dnn_per_link((M * K ,13+ M*K), N_rf)
    # compute interference from k,i
    output_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0 * N_rf / M / K)
    # raw_out_put_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0 / M / K)
    # raw_out_put_0 = tf.tile(tf.expand_dims(raw_out_put_0, axis=3), (1, 1, 1, N_rf))
    # raw_out_put_0 = tf.keras.layers.Reshape((K*M, N_rf))(raw_out_put_0)
    input_i = input_modder(output_0, input_mod, k - 1.0)
    # input_i = input_modder(output_0, input_mod, k - 1.0)
    raw_out_put_i = dnns(input_i)
    raw_out_put_i = sm(raw_out_put_i) # (None, K*M, Nrf)
    # raw_out_put_i = sigmoid((raw_out_put_i - 0.4) * 20.0)
    # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
    out_put_i = tf.reduce_sum(raw_out_put_i, axis=2) # (None, K*M)
    output = [tf.expand_dims(out_put_i, axis=1), tf.expand_dims(raw_out_put_i, axis=1)]
    # begin the second - kth iteration
    for times in range(1, k):
        out_put_i = tf.keras.layers.Reshape((K, M))(out_put_i)
        # input_mod_temp = tf.multiply(out_put_i, input_mod) + input_mod
        input_i = input_modder(out_put_i, input_mod, k - times - 1.0)
        # input_i = input_modder(out_put_i, input_mod, k - times - 1.0)
        raw_out_put_i = dnns(input_i)
        raw_out_put_i = sm(raw_out_put_i)
        # raw_out_put_i = sigmoid((raw_out_put_i - 0.4) * 20.0)
        # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
        out_put_i = tf.reduce_sum(raw_out_put_i, axis=2)
        output[0] = tf.concat([output[0], tf.expand_dims(out_put_i, axis=1)], axis=1)
        output[1] = tf.concat([output[1], tf.expand_dims(raw_out_put_i, axis=1)], axis=1)
    model = Model(inputs, output)
    return model
def FDD_per_link_archetecture_more_G_distillation(M, K, k=2, N_rf=3, output_all=False):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.square(tf.abs(inputs))
    norm = tf.reduce_max(tf.keras.layers.Reshape((K*M, ))(input_mod), axis=1, keepdims=True)
    input_mod = tf.divide(input_mod, tf.expand_dims(norm, axis=1))
    # input_mod = tf.keras.layers.BatchNormalization()(input_mod)
    input_modder = Per_link_Input_modification_most_G_sigmoid(K, M, N_rf, k)
    # input_modder = Per_link_Input_modification_learnable_G(K, M, N_rf, k)
    dnns = dnn_per_link((M * K ,13+ M*K), 1)
    # compute interference from k,i
    output_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0 * N_rf / M / K)
    input_i = input_modder(output_0, input_mod, k - 1.0)
    # input_i = input_modder(output_0, input_mod, k - 1.0)
    raw_out_put_i = dnns(input_i)
    out_put_i = sigmoid(raw_out_put_i) # (None, K*M, Nrf)
    # raw_out_put_i = sigmoid((raw_out_put_i - 0.4) * 20.0)
    # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
    output = [tf.expand_dims(out_put_i, axis=1)]
    # begin the second - kth iteration
    for times in range(1, k):
        out_put_i = tf.keras.layers.Reshape((K, M))(out_put_i)
        # input_mod_temp = tf.multiply(out_put_i, input_mod) + input_mod
        input_i = input_modder(out_put_i, input_mod, k - times - 1.0)
        # input_i = input_modder(out_put_i, input_mod, k - times - 1.0)
        raw_out_put_i = dnns(input_i)
        out_put_i = sigmoid(raw_out_put_i)
        # raw_out_put_i = sigmoid((raw_out_put_i - 0.4) * 20.0)
        # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
        output[0] = tf.concat([output[0], tf.expand_dims(out_put_i, axis=1)], axis=1)
        print(output[0].shape)
    model = Model(inputs, output)
    return model
def FDD_per_link_archetecture_more_G_temperature(M, K, k=2, N_rf=3, output_all=False):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.square(tf.abs(inputs))
    norm = tf.reduce_max(tf.keras.layers.Reshape((K*M, ))(input_mod), axis=1, keepdims=True)
    input_mod = tf.divide(input_mod, tf.expand_dims(norm, axis=1))
    # input_mod = tf.keras.layers.BatchNormalization()(input_mod)
    input_modder = Per_link_Input_modification_most_G(K, M, N_rf, k)
    # input_modder = Per_link_Input_modification_learnable_G(K, M, N_rf, k)
    dnns = dnn_per_link((M * K ,13+ M*K), N_rf)
    # compute interference from k,i
    output_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0 * N_rf / M / K)
    input_i = input_modder(output_0, input_mod, k - 1.0)
    raw_out_put_i = dnns(input_i)
    raw_out_put_i = tf.keras.layers.Softmax(axis=1)(raw_out_put_i) # (None, K*M, Nrf)
    # raw_out_put_i = sigmoid((raw_out_put_i - 0.4) * 20.0)
    # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
    out_put_i = tf.reduce_sum(raw_out_put_i, axis=2) # (None, K*M)
    output = [tf.expand_dims(out_put_i, axis=1), tf.expand_dims(raw_out_put_i, axis=1)]
    # begin the second - kth iteration
    for times in range(1, k):
        out_put_i = tf.keras.layers.Reshape((K, M))(out_put_i)
        # input_mod_temp = tf.multiply(out_put_i, input_mod) + input_mod
        input_i = input_modder(out_put_i, input_mod, k - times - 1.0)
        raw_out_put_i = dnns(input_i)
        raw_out_put_i = tf.keras.layers.Softmax(axis=1)(tf.math.scalar_mul(1.0/(1.0-times/(k-1)*0.9), raw_out_put_i))
        # raw_out_put_i = sigmoid((raw_out_put_i - 0.4) * 20.0)
        # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
        out_put_i = tf.reduce_sum(raw_out_put_i, axis=2)
        output[0] = tf.concat([output[0], tf.expand_dims(out_put_i, axis=1)], axis=1)
        output[1] = tf.concat([output[1], tf.expand_dims(raw_out_put_i, axis=1)], axis=1)
    model = Model(inputs, output)
    return model
def FDD_one_at_a_time(M, K, k=2, N_rf=3, output_all=False):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.square(tf.abs(inputs))
    norm = tf.reduce_max(tf.keras.layers.Reshape((K * M,))(input_mod), axis=1, keepdims=True)
    input_mod = tf.divide(input_mod, tf.expand_dims(norm, axis=1))
    # input_modder = Per_link_sequential_modification_compressedX(K, M, N_rf, 1)
    input_modder = Sequential_Per_link_Input_modification_most_G_raw_self(K, M, N_rf, N_rf)
    dnn_model = dnn_sequential((K*M, 17 + N_rf))
    output_final = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :])) # inital output/planning
    input_i = input_modder(output_final, input_mod, N_rf - 1.0)
    raw_out_put_i = dnn_model(input_i)
    out_put_i = tf.keras.layers.Softmax(axis=1)(raw_out_put_i)[:, :, 0]  # (None, K*M)
    output = [tf.expand_dims(out_put_i, axis=1)]
    output_final = tf.keras.layers.Reshape((K * M,))(out_put_i)
    # begin the second - Nrf_th iteration
    for times in range(1, N_rf):
        out_put_i = tf.keras.layers.Reshape((K, M))(output_final)
        # input_mod_temp = tf.multiply(out_put_i, input_mod) + input_mod
        input_i = input_modder(out_put_i, input_mod, N_rf - times - 1.0)
        raw_out_put_i = dnn_model(input_i)
        out_put_i = tf.keras.layers.Softmax(axis=1)(raw_out_put_i)[:, :, 0]
        # raw_out_put_i = sigmoid((raw_out_put_i - 0.4) * 20.0)
        # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
        output_final = output_final + out_put_i
        output[0] = tf.concat([output[0], tf.expand_dims(out_put_i, axis=1)], axis=1)
    output.append(output_final)
    model = Model(inputs, output)
    return model
def Input_normalization_all(raw_input):
    K = raw_input.shape[1]
    M = raw_input.shape[2]
    # mean = tf.reduce_mean(tf.keras.layers.Reshape((K * M,))(raw_input), axis=1, keepdims=True)
    # std = tf.math.reduce_std(tf.keras.layers.Reshape((K * M,))(raw_input), axis=1, keepdims=True)
    # input_mod = tf.divide(raw_input - tf.expand_dims(mean, axis=1), tf.expand_dims(std, axis=1))
    max = tf.reduce_max(tf.keras.layers.Reshape((K * M,))(raw_input), axis=1, keepdims=True)
    input_mod = tf.divide(raw_input, tf.expand_dims(max, axis=1))
    avg_max = tf.reduce_mean(max)
    return input_mod, avg_max
def Input_normalization_per_user(raw_input, avg_max=None):
    # mean = tf.reduce_mean(raw_input, axis=2, keepdims=True)
    # std = tf.math.reduce_std(raw_input, axis=2, keepdims=True)
    # input_mod = tf.divide(raw_input - mean, std)
    if avg_max is None:
        max = tf.reduce_max(raw_input, axis=2, keepdims=True)
        avg_max = tf.reduce_mean(max)
    input_mod = tf.divide(raw_input, avg_max)
    return input_mod, avg_max
def FDD_one_at_a_time_iterable(M, K, k=2, N_rf=3, normalization=True, avg_max=None):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.abs(inputs)
    if normalization:
        input_mod = tf.square(tf.divide(input_mod, avg_max))
    else:
        input_mod = tf.square(input_mod)
    input_modder = Sequential_Per_link_Input_modification_most_G_raw_self(K, M, N_rf, 2*N_rf)
    dnn_model = dnn_sequential((K*M, 9 + 2*N_rf))
    output_final = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0) # inital output/planning
    input_i = input_modder(output_final, input_mod, 2*N_rf - 1.0)
    raw_out_put_i = dnn_model(input_i)
    out_put_i = tf.keras.layers.Softmax(axis=1)(raw_out_put_i)[:, :, 0]  # (None, K*M)
    output = [tf.expand_dims(out_put_i, axis=1)]
    output_final = tf.keras.layers.Reshape((K * M,))(out_put_i)

    # begin the second - Nrf_th iteration
    for times in range(1, 2*N_rf):
        if times < N_rf:
            out_put_i = tf.keras.layers.Reshape((K, M))(output_final)
        else:
            output_final = tf.reduce_sum(output[0][:, (-N_rf+1):], axis=1)
            out_put_i = tf.keras.layers.Reshape((K, M))(output_final)
        # input_mod_temp = tf.multiply(out_put_i, input_mod) + input_mod
        input_i = input_modder(out_put_i, input_mod, 2*N_rf - times - 1.0)
        raw_out_put_i = dnn_model(input_i)
        out_put_i = tf.keras.layers.Softmax(axis=1)(raw_out_put_i)[:, :, 0]
        # raw_out_put_i = sigmoid((raw_out_put_i - 0.4) * 20.0)
        # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
        output_final = output_final + out_put_i
        output[0] = tf.concat([output[0], tf.expand_dims(out_put_i, axis=1)], axis=1)
    output.append(output_final)
    model = Model(inputs, output)
    return model
def FDD_per_link_archetecture_more_G_no_SM_between_passes(M, K, k=2, N_rf=3, output_all=False):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.square(tf.abs(inputs))
    input_mod = tf.keras.layers.BatchNormalization()(input_mod)
    input_modder = Per_link_Input_modification_compress_XG_alt(K, M, N_rf, k)
    dnns = dnn_per_link((M * K,9 + 2*K + 64), N_rf)
    # compute interference from k,i
    output_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0 * N_rf / M / K)
    input_i = input_modder(output_0, input_mod, k - 1.0)
    raw_out_put_i = dnns(input_i)
    sm_raw_out_put_i = tf.keras.layers.Softmax(axis=1)(raw_out_put_i) # (None, K*M, Nrf)

    raw_out_put_i = tf.reduce_sum(raw_out_put_i, axis=2)
    # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
    out_put_i = tf.reduce_sum(sm_raw_out_put_i, axis=2) # (None, K*M)
    output = [tf.expand_dims(out_put_i, axis=1), tf.expand_dims(sm_raw_out_put_i, axis=1)]
    # begin the second - kth iteration
    for times in range(1, k):
        raw_out_put_i = tf.keras.layers.Reshape((K, M))(sigmoid(raw_out_put_i))
        input_i = input_modder(raw_out_put_i, input_mod, k - times - 1.0)
        raw_out_put_i = dnns(input_i)
        sm_raw_out_put_i = tf.keras.layers.Softmax(axis=1)(raw_out_put_i)  # (None, K*M, Nrf)
        raw_out_put_i = tf.reduce_sum(raw_out_put_i, axis=2)
        # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
        out_put_i = tf.reduce_sum(sm_raw_out_put_i, axis=2)
        output[0] = tf.concat([output[0], tf.expand_dims(out_put_i, axis=1)], axis=1)
        output[1] = tf.concat([output[1], tf.expand_dims(sm_raw_out_put_i, axis=1)], axis=1)
    model = Model(inputs, output)
    return model
def FDD_per_link_archetecture(M, K, k=2, N_rf=3, output_all=False):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.square(tf.abs(inputs))
    input_mod = tf.keras.layers.BatchNormalization()(input_mod)
    input_modder = Interference_Input_modification(K, M, N_rf, k)
    dnns = dnn_per_link((M * K, 4 + M * K), N_rf)
    # compute interference from k,i
    output_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0 * N_rf / M / K)
    input_i = input_modder(output_0, input_mod, k - 1.0)
    out_put_i = dnns(input_i)
    out_put_i = tf.keras.layers.Softmax(axis=1)(out_put_i)
    # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
    out_put_i = tf.reduce_sum(out_put_i, axis=2)
    if output_all:
        output_0 = tf.keras.layers.Reshape((1, M * K))(out_put_i)
    # begin the second - kth iteration
    for times in range(1, k):
        out_put_i = tf.keras.layers.Reshape((K, M))(out_put_i)
        input_i = input_modder(out_put_i, input_mod, k - times - 1.0)
        out_put_i = dnns(input_i)
        out_put_i = tf.keras.layers.Softmax(axis=1)(out_put_i)
        # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
        out_put_i = tf.reduce_sum(out_put_i, axis=2)
        if output_all:
            output_0 = tf.concat((output_0, tf.keras.layers.Reshape((1, M * K))(out_put_i)), axis=1)
    model = Model(inputs, out_put_i)
    if output_all:
        model = Model(inputs, output_0)
    return model
def FDD_per_link_archetecture_sigmoid(M, K, k=2, N_rf=3, output_all=False):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.square()
    input_modder = Interference_Input_modification(K, M, N_rf, k)
    dnns = dnn_per_link((M * K, 4 + M * K), N_rf)
    # compute interference from k,i
    output_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]))
    input_i = input_modder(output_0, input_mod, k - 1.0)
    out_put_i = dnns(input_i)
    out_put_i = sigmoid(tf.reduce_sum(out_put_i, axis=2))
    # begin the second - kth iteration
    if output_all:
        output_0 = tf.keras.layers.Reshape((1, M * K))(out_put_i)
    for times in range(1, k):
        out_put_i = tf.keras.layers.Reshape((K, M))(out_put_i)
        input_i = input_modder(out_put_i, input_mod, k - times - 1.0)
        out_put_i = dnns(input_i)
        out_put_i = sigmoid(tf.reduce_sum(out_put_i, axis=2))
        if output_all:
            output_0 = tf.concat((output_0, tf.keras.layers.Reshape((1, M * K))(out_put_i)), axis=1)
    model = Model(inputs, out_put_i)
    if output_all:
        model = Model(inputs, output_0)
    return model
def FDD_Dumb_model(M, K, k=2, N_rf=3):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.square(tf.abs(inputs))
    input_mod = tf.keras.layers.Reshape((K * M,))(input_mod)
    output_0 = tf.stop_gradient(tf.multiply(tf.zeros((K * M,)), input_mod[:, :]) + 1.0 * N_rf / M * K)
    input_pass_0 = tf.keras.layers.Concatenate(axis=1)((output_0, input_mod))
    # dnn_model = DNN_3_layer_model((3*K*M), M, K, 0)
    dnn_model = DNN_3_layer_Thicc_model((2 * K * M), M, K, Nrf=N_rf)
    output_i = tf.keras.layers.Reshape((N_rf, K * M))(dnn_model(input_pass_0))
    output_i = tf.reduce_sum(tf.keras.layers.Softmax(axis=2)(output_i), axis=1)
    for i in range(1, k):
        input_pass_i = tf.keras.layers.Concatenate(axis=1)((output_i, input_mod))
        output_i = tf.keras.layers.Reshape((N_rf, K * M))(dnn_model(input_pass_i))
        output_i = tf.reduce_sum(tf.keras.layers.Softmax(axis=2)(output_i), axis=1)
    model = Model(inputs, output_i)
    return model
def per_row_DNN(input_shape, M, N_rf=1, name="dnn"):
    inputs = Input(shape=input_shape)
    x = Dense(256)(inputs)
    x = sigmoid(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = Dense(64)(x)
    # x = sigmoid(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = Dense(N_rf, bias_initializer="ones")(x)
    model = Model(inputs, x, name=name)
    return model
def tiny_DNN(input_shape, N_rf):
    inputs = Input(shape=input_shape, dtype=tf.float32)
    x = Dense(128)(inputs)
    x = sigmoid(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Dense(N_rf, bias_initializer="ones")(x)
    model = Model(inputs, x)
    return model
def LSTM_like_model_for_FDD(M, K, N_rf, k):
    inputs = Input(shape=(K, M), dtype=tf.float32)
    input_modder = Interference_Input_modification(K, M, N_rf, k)
    dnn1 = tiny_DNN((M * K, 4 + M * K), N_rf)
    dnn2 = tiny_DNN((M * K, 4 + M * K), N_rf)
    dnn3 = tiny_DNN((M * K, 4 + M * K), N_rf)
    dnn4 = tiny_DNN((M * K, 4 + M * K), N_rf)
    output_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), inputs[:, :, :]) + 1.0 * N_rf / M / K)
    input_i = input_modder(output_0, inputs, k - 1.0)
    state_0 = tf.tile(tf.keras.layers.Reshape((K * M, 1))(output_0), (1, 1, N_rf))
    x = tf.multiply(sigmoid(dnn1(input_i)), state_0)  # forget gate
    state_i = x + tf.multiply(sigmoid(dnn2(input_i)), tf.tanh(dnn3(input_i)))
    output_i = tf.multiply(sigmoid(dnn4(input_i)), state_i)
    output_i = tf.reduce_sum(tf.keras.layers.Softmax(axis=1)(output_i), axis=2)
    output = [tf.expand_dims(output_i, axis=1)]
    for i in range(1, k):
        input_i = input_modder(tf.keras.layers.Reshape((K, M))(output_i), inputs, k - 1.0 - i)
        x = tf.multiply(sigmoid(dnn1(input_i)), state_i)  # forget gate
        state_i = x + tf.multiply(sigmoid(dnn2(input_i)), tf.tanh(dnn3(input_i)))
        output_i = tf.multiply(sigmoid(dnn4(input_i)), state_i)
        output_i = tf.reduce_sum(tf.keras.layers.Softmax(axis=1)(output_i), axis=2)
        output.append(tf.expand_dims(output_i, axis=1))
    output = tf.concat(output, axis=1)
    model = Model(inputs, output)
    return model
def FDD_per_user_architecture_double_softmax_all_softmaxes(M, K, k=2, N_rf=3, output_all=False):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.square(tf.abs(inputs))  # (None, K, M)
    input_modder = Interference_Input_modification_per_user(K, M, N_rf, k)
    sm = tf.keras.layers.Softmax()
    dnn = per_user_DNN((K, M * K + M + 3), M, N_rf)
    decision_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0 * N_rf / M / K)
    input_pass_0 = input_modder(decision_0, input_mod, k - 1.0)
    output_i = dnn(input_pass_0)
    selection_i = tf.reduce_sum(tf.keras.layers.Softmax(axis=1)(output_i[:, :, -N_rf:]), axis=2)
    output_i = tf.multiply(sm(output_i[:, :, :-N_rf]), tf.tile(tf.expand_dims(selection_i, axis=2), (1, 1, M)))
    if output_all:
        output_0 = tf.keras.layers.Reshape((1, M * K))(output_i)
    tim = []
    for times in range(1, k):
        output_i = tf.keras.layers.Reshape((K, M))(output_i)
        input_i = input_modder(output_i, input_mod, k - times - 1.0)
        output_i = dnn(input_i)
        selection_i = tf.reduce_sum(tf.keras.layers.Softmax(axis=1)(output_i[:, :, -N_rf:]), axis=2)
        if times == k - 1:
            tim = [sm(output_i[:, :, :-N_rf]), selection_i]
        output_i = tf.multiply(sm(output_i[:, :, :-N_rf]), tf.tile(tf.expand_dims(selection_i, axis=2), (1, 1, M)))
        if output_all:
            output_0 = tf.concat((output_0, tf.keras.layers.Reshape((1, M * K))(output_i)), axis=1)
    output_i = tf.keras.layers.Reshape((K * M,))(output_i)
    model = Model(inputs, [output_i] + tim)
    if output_all:
        model = Model(inputs, [output_0] + tim)
    return model
def FDD_per_user_architecture_return_all_softmaxes(M, K, k=2, N_rf=3, yes_abs=False):
    if yes_abs:
        inputs = Input(shape=(K, M), dtype=tf.complex64)
        input_mod = tf.square(tf.abs(inputs))  # (None, K, M)
    else:
        inputs = Input(shape=(K, M), dtype=tf.float32)
        input_mod = inputs
    input_modder = Interference_Input_modification_per_user(K, M, N_rf, k)
    sm = tf.keras.layers.Softmax()
    dnn = per_user_DNN((K, M * K + M + 3), M, N_rf)
    decision_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0 * N_rf/M/K)
    input_pass_0 = input_modder(decision_0, input_mod, k - 1.0)
    output_i = dnn(input_pass_0)
    selection_i = tf.reduce_sum(tf.keras.layers.Softmax(axis=1)(output_i[:, :, -N_rf:]), axis=2)
    per_user_selection_i = sm(output_i[:, :, :-N_rf])
    output_i = tf.multiply(per_user_selection_i, tf.tile(tf.expand_dims(selection_i, axis=2), (1, 1, M)))
    output = [tf.keras.layers.Reshape((1, M * K))(output_i), tf.expand_dims(selection_i, axis=1)
        , tf.expand_dims(per_user_selection_i, axis=1)]
    for times in range(1, k):
        output_i = tf.keras.layers.Reshape((K, M))(output_i)
        input_i = input_modder(output_i, input_mod, k - times - 1.0)
        output_i = dnn(input_i)
        selection_i = tf.reduce_sum(tf.keras.layers.Softmax(axis=1)(output_i[:, :, -N_rf:]), axis=2)
        per_user_selection_i = sm(output_i[:, :, :-N_rf])
        output_i = tf.multiply(per_user_selection_i, tf.tile(tf.expand_dims(selection_i, axis=2), (1, 1, M)))

        output[0] = tf.concat((output[0], tf.keras.layers.Reshape((1, M * K))(output_i)), axis=1)
        output[1] = tf.concat((output[1], tf.expand_dims(selection_i, axis=1)), axis=1)
        output[2] = tf.concat((output[2], tf.expand_dims(per_user_selection_i, axis=1)), axis=1)
    model = Model(inputs, output)
    return model
def FDD_per_user_architecture(M, K, k=2, N_rf=3):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.square(tf.abs(inputs))  # (None, K, M)
    input_modder = Interference_Input_modification_per_user(K, M, N_rf, k)
    sm = tf.keras.layers.Softmax()
    dnn = per_user_DNN((K, M * K + M + 3), M)
    decision_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0 * N_rf / M / K)
    input_pass_0 = input_modder(decision_0, input_mod, k - 1.0)
    output_i = dnn(input_pass_0)
    output_i = tf.multiply(sm(output_i[:, :, :-1]), tf.expand_dims(sigmoid(output_i[:, :, -1]), axis=2))
    for times in range(1, k):
        output_i = tf.keras.layers.Reshape((K, M))(output_i)
        input_i = input_modder(output_i, input_mod, k - times - 1.0)
        output_i = dnn(input_i)
        output_i = tf.multiply(sm(output_i[:, :, :-1]), tf.expand_dims(sigmoid(output_i[:, :, -1]), axis=2))
    output_i = tf.keras.layers.Reshape((K * M,))(output_i)
    model = Model(inputs, output_i)
    return model
def FDD_per_user_architecture_double_softmax(M, K, k=2, N_rf=3, output_all=False):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.square(tf.abs(inputs))  # (None, K, M)
    input_modder = Interference_Input_modification_per_user(K, M, N_rf, k)
    sm = tf.keras.layers.Softmax()
    dnn = per_user_DNN((K, M * K + M + 3), M, N_rf)
    decision_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), input_mod[:, :, :]) + 1.0 * N_rf / M / K)
    input_pass_0 = input_modder(decision_0, input_mod, k - 1.0)
    output_i = dnn(input_pass_0)
    selection_i = tf.reduce_sum(tf.keras.layers.Softmax(axis=1)(output_i[:, :, -N_rf:]), axis=2)
    output_i = tf.multiply(sm(output_i[:, :, :-N_rf]), tf.tile(tf.expand_dims(selection_i, axis=2), (1, 1, M)))
    if output_all:
        output_0 = tf.keras.layers.Reshape((1, M * K))(output_i)
    for times in range(1, k):
        output_i = tf.keras.layers.Reshape((K, M))(output_i)
        input_i = input_modder(output_i, input_mod, k - times - 1.0)
        output_i = dnn(input_i)
        selection_i = tf.reduce_sum(tf.keras.layers.Softmax(axis=1)(output_i[:, :, -N_rf:]), axis=2)
        output_i = tf.multiply(sm(output_i[:, :, :-N_rf]), tf.tile(tf.expand_dims(selection_i, axis=2), (1, 1, M)))
        if output_all:
            output_0 = tf.concat((output_0, tf.keras.layers.Reshape((1, M * K))(output_i)), axis=1)
    output_i = tf.keras.layers.Reshape((K * M,))(output_i)
    model = Model(inputs, output_i)
    if output_all:
        model = Model(inputs, output_0)
    return model
def FDD_per_link_LSTM(M, K, k=2, N_rf=3, output_all=False):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.square(tf.abs(inputs))
    input_mod = tf.keras.layers.BatchNormalization()(input_mod)
    lstm_module = LSTM_like_model_for_FDD(M, K, N_rf, k)
    out = lstm_module(input_mod)
    # compute interference from k,i
    model = Model(inputs, out[:, -1])
    if output_all:
        model = Model(inputs, out)
    return model
class NN_Clustering():
    def __init__(self, cluster_count, original_dim, reduced_dim=10):
        self.cluster_count = cluster_count
        self.original_dim = original_dim
        self.reduced_dim = reduced_dim
        self.cluster_mean = None
        self.assignment = np.zeros((cluster_count,))
        self.decoder = self.decoder_network((reduced_dim,))
        self.encoder = self.encoder_network((original_dim,))
        self.optimizer = tf.optimizers.Adam()

    def encoder_network(self, input_shape):
        # instead of feeding in (epochs, K, M), feed in (epochs*K, M) instead
        inputs = Input(shape=input_shape)
        x = Dense(self.original_dim * self.reduced_dim)(inputs)
        x = LeakyReLU()(x)
        x = LeakyReLU(self.original_dim * self.reduced_dim)(x)
        x = LeakyReLU()(x)
        x = LeakyReLU(self.original_dim * self.reduced_dim)(x)
        x = LeakyReLU()(x)
        x = Dense(self.reduced_dim)(x)
        # x = sigmoid(x)
        model = Model(inputs, x, name="K_mean_encoder")
        return model

    def decoder_network(self, input_shape):
        # instead of feeding in (epochs, K, M), feed in (epochs*K, M) instead
        inputs = Input(shape=input_shape)
        x = Dense(self.original_dim)(inputs)
        x = LeakyReLU()(x)
        x = LeakyReLU(self.original_dim * self.reduced_dim)(x)
        x = LeakyReLU()(x)
        x = LeakyReLU(self.original_dim * self.reduced_dim)(x)
        x = LeakyReLU()(x)
        x = Dense(self.original_dim)(x)
        # x = sigmoid(x)
        model = Model(inputs, x, name="K_mean_decoder")
        return model

    def train_network(self, G):
        N = 10000
        train_history = np.zeros((N, 2))
        # only care about the absolute value
        G = np.abs(G)
        # normalize all the G values
        for n in range(0, G.shape[0]):
            for k in range(0, G.shape[1]):
                G[n, k] = (G[n, k] - G[n, k].min()) / (G[n, k].max() - G[n, k].min())
        # somehow flatten G
        data = tf.reshape(G, (G.shape[0] * K, M))
        # init K mean algo for assignments and cluster mean
        self.assignment = np.zeros((self.cluster_count, G.shape[0] * K)).astype(np.float32)
        clustering_param = self.encoder(data)
        self.cluster_mean = clustering_param[0:self.cluster_count].numpy().T
        for i in range(0, G.shape[0]):
            self.assignment[np.random.randint(0, self.cluster_count - 1), i] = 1
        for i in range(N):
            with tf.GradientTape() as tape:
                clustering_param = self.encoder(data)
                recovered_param = self.decoder(clustering_param)
                loss1 = tf.keras.losses.MeanSquaredError()(tf.abs(data), recovered_param)
                loss2 = tf.keras.losses.MeanSquaredError()(clustering_param, (self.cluster_mean @ self.assignment).T)
                loss = loss1 + loss2
            variables = self.encoder.trainable_variables + self.decoder.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
            train_history[i, 0] = loss1
            train_history[i, 1] = loss2
            print(loss1, loss2)
            if i % 100 == 0 and i >= 200:
                improvement = train_history[i - (100 * 2): i - 100, 0].mean() - train_history[
                                                                                i - 100: i,
                                                                                0].mean()
                if improvement <= 0.01:
                    break

            self.train_k_means_step(clustering_param)

    def train_k_means_step(self, clustering_param):
        # update assignments
        points_expanded = tf.expand_dims(clustering_param, 0)
        points_expanded = tf.tile(points_expanded, [self.cluster_count, 1, 1])
        centroids_expanded = tf.expand_dims(self.cluster_mean.T, 1)
        centroids_expanded = tf.tile(centroids_expanded, [1, clustering_param.shape[0], 1])
        distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
        assignments = tf.argmin(distances, axis=0)
        self.assignment = np.zeros((self.cluster_count, clustering_param.shape[0])).astype(np.float32)
        for i in range(clustering_param.shape[0]):
            self.assignment[assignments[i], i] = 1
        means = []
        for c in range(self.cluster_count):
            means.append(tf.reduce_mean(
                tf.gather(clustering_param,
                          tf.reshape(
                              tf.where(
                                  tf.equal(assignments, c)
                              ), [1, -1])
                          ), axis=1))
        new_centroids = tf.concat(means, 0)
        self.cluster_mean = new_centroids.numpy().T.astype(np.float32)

    def save_model(self, dir):
        try:
            os.mkdir(dir)
        except:
            print("already_exist")
        encoder_template = dir + "encoder.h5"
        decoder_template = dir + "decoder.h5"
        means_template = dir + "centroid.npy"
        self.encoder.save(encoder_template)
        self.decoder.save(decoder_template)
        np.save(means_template, self.cluster_mean)

    def load_model(self, dir):
        encoder_template = dir + "encoder.h5"
        decoder_template = dir + "decoder.h5"
        means_template = dir + "centroid.npy"
        self.encoder = tf.keras.models.load_model(encoder_template)
        self.decoder = tf.keras.models.load_model(decoder_template)
        self.cluster_mean = np.load(means_template)

    def evaluate(self, G):
        G = tf.abs(G).numpy()
        G_original = G.copy()
        for n in range(0, G.shape[0]):
            for k in range(0, G.shape[1]):
                G[n, k] = (G[n, k] - G[n, k].min()) / (G[n, k].max() - G[n, k].min())
        output = np.zeros((G.shape[0], G.shape[1] * G.shape[2]))
        # kmeans_tot = KMeans(n_clusters=N_rf, random_state=0).fit(G[0:500].reshape(500*G.shape[1], G.shape[2]))
        for n in range(0, G.shape[0]):
            # somehow flatten G
            clustering_param = self.encoder(G[n])
            points_expanded = tf.expand_dims(clustering_param, 0)
            points_expanded = tf.tile(points_expanded, [self.cluster_count, 1, 1])
            centroids_expanded = tf.expand_dims(self.cluster_mean.T, 1)
            centroids_expanded = tf.tile(centroids_expanded, [1, clustering_param.shape[0], 1])
            distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
            assignments = tf.argmin(distances, axis=0)
            clusters = []
            for i in range(0, self.cluster_count):
                clusters.append(np.zeros(G[0].shape))
            for i in range(G.shape[1]):
                clusters[assignments[i]][i] = (G_original[n, i])
            for i in range(0, self.cluster_count):
                max = int(np.argmax(clusters[i]))
                if np.sum(clusters[i]) != 0:
                    output[n, max] = 1
        return output
def distributed_DNN(input_shape, N_rf):
    inputs = Input(shape=input_shape)
    x = Dense(64)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = sigmoid(x)
    x = Dense(64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = sigmoid(x)
    x = Dense(N_rf)(x)
    # x = sigmoid(x)
    model = Model(inputs, x)
    return model
def FDD_reduced_output_space(M, K, N_rf=3):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.square(tf.abs(inputs))
    norm = tf.reduce_max(tf.keras.layers.Reshape((K * M,))(input_mod), axis=1, keepdims=True)
    input_modder = Reduced_output_input_mod(K, M, N_rf,3)
    precoder_input_modder = Reduced_output_precoder_input_mod(K, M, N_rf,3)
    input_mod = tf.divide(input_mod, tf.tile(tf.expand_dims(norm, axis=1), (1, K, M)))
    user_selection_dnn = per_row_DNN((K, 3*M + M + 1), M, N_rf, name="per_user_dnn")
    precoder_selection_dnn = per_row_DNN((N_rf*M, 4+M), M, 1, name="per_precoder_dnn")
    user_selection = user_selection_dnn(input_modder(input_mod, K))
    user_selection = tf.keras.layers.Softmax(axis=1)(user_selection)
    # user_selection = user_selection + tf.stop_gradient(user_selection / tf.reduce_max(user_selection, axis=1, keepdims=True) - user_selection)
    # user_selection = user_selection + tf.stop_gradient(tf.divide(user_selection, tf.tile(tf.reduce_max(user_selection, axis=1, keepdims=True), [1, K, 1]))-user_selection)
    user_selection = tf.transpose(user_selection, perm=[0, 2, 1])
    input_mod = tf.matmul(user_selection, input_mod)
    precoder_selection = precoder_selection_dnn(precoder_input_modder(None, input_mod))
    precoder_selection = tf.keras.layers.Reshape((N_rf, M))(precoder_selection)
    precoder_selection = tf.keras.layers.Softmax(axis=2)(precoder_selection)
    user_selection = tf.reduce_sum(user_selection, axis=1, keepdims=True)
    # precoder_selection = precoder_selection + tf.stop_gradient(tf.divide(precoder_selection, tf.reduce_max(precoder_selection, axis=1, keepdims=True))-precoder_selection)
    precoder_selection = tf.reduce_sum(precoder_selection, axis=1, keepdims=True)
    user_selection = tf.transpose(user_selection, perm=[0, 2, 1])
    out = tf.matmul(user_selection, precoder_selection)
    out = tf.keras.layers.Reshape((M*K, ))(out)
    model = Model(inputs, out)
    return model
def FDD_RNN_model(M, K, N_rf=3):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.square(tf.abs(inputs))
    emb_size = K+M
    # lstm_1 = tf.keras.layers.LSTM(256, return_state=True)
    lstm_1 = tf.keras.layers.RNN(tf.keras.layers.GRUCell(emb_size), return_state=True)
    lstm_2 = tf.keras.layers.RNN(tf.keras.layers.GRUCell(emb_size), return_state=True)
    lstm_out = tf.keras.layers.RNN(tf.keras.layers.GRUCell(emb_size), return_sequences=True)
    out_1 = lstm_1(input_mod)
    out_2 = lstm_2(tf.transpose(input_mod, perm=[0,2,1]), initial_state=out_1[1])
    out_1_re = lstm_1(input_mod, initial_state=out_2[1])
    out_2_re = lstm_2(tf.transpose(input_mod, perm=[0,2,1]), initial_state=out_1_re[1])
    output_0 = tf.stop_gradient(tf.multiply(tf.zeros((N_rf, N_rf)), input_mod[:, :N_rf, :N_rf]) + tf.eye(N_rf))
    out = lstm_out(output_0, initial_state = out_2_re[1])
    x = Dense(512)(out)
    x = sigmoid(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x_k = Dense(K)(x)
    x_k = tf.keras.layers.Softmax(axis=-1)(x_k)
    output_k = tf.reduce_sum(x_k, axis=1)
    x_m = Dense(M)(x)
    x_m = tf.keras.layers.Softmax(axis=-1)(x_m)
    output_m = tf.reduce_sum(x_m, axis=1)
    output_k = tf.expand_dims(output_k, axis=2)
    output_m = tf.expand_dims(output_m, axis=1)
    output_square = tf.matmul(output_k, output_m)
    output = tf.keras.layers.Reshape((K*M, ))(output_square)
    return Model(inputs, output, name="LSTM_MODEL")


def Top2Precoder_model(M, K, k=2, N_rf=3, filter=2):
    smol_input = Input(shape=(K, filter), dtype=tf.float32, name="smol_input")
    position_matrix = Input(shape=(filter*K, M), dtype=tf.float32, name="position_matrix")
    inputs = Input(shape=(K, M), dtype=tf.complex64, name="Main_inputs")
    # utilized layers
    input_modder = TopPrecoderPerUserInputMod(K, M, N_rf, k, filter)
    output_stretcher = X_extends(K, M, N_rf, filter)
    output_flatten = tf.keras.layers.Reshape((K*M, ), name="output_flatten")
    dnns = dnn_per_link((filter * K, 7 + filter * K), N_rf)
    sm = tf.keras.layers.Softmax(axis=1, name="raw_output_i_softmax")
    # normalizing inputs
    input_mod = tf.abs(inputs, name="abs1")
    norm = tf.reduce_max(tf.keras.layers.Reshape((K * M,), name="max1")(input_mod), axis=1, keepdims=True)
    input_mod = tf.divide(input_mod, tf.expand_dims(norm, axis=1), name="div1")
    smol_input_mod = tf.divide(smol_input, tf.expand_dims(norm, axis=1), name="div2")
    # initial null input
    output_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, filter), name="mul1"), smol_input_mod) + 1.0*N_rf/K/filter)
    output_0 = tf.stop_gradient(tf.keras.layers.Reshape((K*filter, ), name="smol_output_0_flatten")(output_0))
    input_i = input_modder(output_stretcher(output_0, position_matrix), output_0, input_mod, smol_input_mod, k - 1.0)
    raw_out_put_i = dnns(input_i)
    raw_out_put_i = sm(raw_out_put_i)  # (None, K*filter, Nrf)
    out_put_i = tf.reduce_sum(raw_out_put_i, axis=2)  # (None, K*filter)
    full_output_i = output_stretcher(out_put_i, position_matrix)
    f_output_0 = tf.expand_dims(output_flatten(full_output_i), axis=1)
    f_output_1 = tf.expand_dims(out_put_i, axis=1)
    # begin the second - kth iteration
    for times in range(1, k):
        input_i = input_modder(full_output_i, out_put_i, input_mod, smol_input_mod, k - times - 1.0)
        # input_i = input_modder(out_put_i, input_mod, k - times - 1.0)
        raw_out_put_i = dnns(input_i)
        raw_out_put_i = sm(raw_out_put_i)
        out_put_i = tf.reduce_sum(raw_out_put_i, axis=2)
        full_output_i = output_stretcher(out_put_i, position_matrix)
        # output[0] = concatter1([output[0], tf.expand_dims(full_output_i, axis=1)])
        # output[1] = concatter2([output[1], tf.expand_dims(out_put_i, axis=1)])
        f_output_0 = tf.expand_dims(output_flatten(full_output_i), axis=1)
        f_output_1 = tf.expand_dims(out_put_i, axis=1)
    # output[0] = concatter(output[0])
    # output[1] = concatter(output[1])
    model = Model(inputs=[inputs, smol_input, position_matrix], outputs=[f_output_0, f_output_1], name="Top2Precoder_model")
    print(model.summary())
    return model


def FDD_distributed_then_general_architecture(M, K, k=2, N_rf=3, output_all=False):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.square(tf.abs(inputs))
    norm = tf.reduce_max(tf.keras.layers.Reshape((K * M,))(input_mod), axis=1, keepdims=True)
    input_mod = tf.divide(input_mod, tf.tile(tf.expand_dims(norm, axis=1), (1, K, M)))
    input_modder = Distributed_input_mod(K, M, N_rf, k)
    dnns = distributed_DNN((M * K, 10), N_rf)
    input_i = input_modder(input_mod)
    raw_out_put_i = dnns(input_i)
    sm_raw_out_put_i = tf.keras.layers.Softmax(axis=1)(raw_out_put_i)  # (None, K*M, Nrf)
    # out_put_i = tfa.layers.Sparsemax(axis=1)(out_put_i)
    sum_sm_raw_out_put_i = tf.reduce_sum(sm_raw_out_put_i, axis=2)  # (None, K*M)
    # output = [tf.expand_dims(out_put_i, axis=1), tf.expand_dims(raw_out_put_i, axis=1)]
    regularizer = sigmoid(Dense(M*K)(tf.keras.layers.Reshape((K*M,))(input_mod)*sum_sm_raw_out_put_i))
    out_put_i = tf.multiply(raw_out_put_i, tf.tile(tf.expand_dims(regularizer, axis=2), (1,1,N_rf)))
    out_put_i = tf.reduce_sum(tf.keras.layers.Softmax(axis=1)(out_put_i), axis=2)
    output = [tf.expand_dims(out_put_i, axis=1)]
    # for i in range(k):
    #     input_i =
    # # begin the second - kth iteration
    # x = Dense(512)(output_after_softmax)
    # x = Dense(3200)(x)
    # output_before_softmax = x * output_before_softmax
    model = Model(inputs, out_put_i)
    return model
def DNN_All_info_scheduler(M, K, Nrf):
    inputs = Input(shape=(Nrf, 2*M*K + Nrf))
    x = Dense(512)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = sigmoid(x)
    x = Dense(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = sigmoid(x)
    x = Dense(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = sigmoid(x)
    x = Dense(M*K)(x)
    x = tf.keras.layers.Softmax(axis=2)(x)
    return Model(inputs, x)
def All_info_scheduler(M, K, N_rf=3):
    input_modder = AllInput_input_mod(K, M, N_rf)
    dnn = DNN_All_info_scheduler(M, K, N_rf)

    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.square(tf.abs(inputs))  # (None, K, M)
    input_mod = tf.keras.layers.BatchNormalization()(input_mod)
    input_mod = tf.keras.layers.Reshape((K*M, ))(input_mod)
    output_0 = tf.stop_gradient(tf.multiply(tf.zeros((K*M, )), input_mod[:, :]))
    input_0 = input_modder(output_0, input_mod)
    raw_output_1 = dnn(input_0)
    for i in range(1, N_rf):
        output_i = tf.reduce_sum(raw_output_1[:, :i, :], axis=1)
        input_i = input_modder(output_i, input_mod)
        raw_output_1 = dnn(input_i)
    output_i = tf.reduce_sum(raw_output_1[:, :, :], axis=1)
    model = Model(inputs, output_i)
    return model
#============================== FDD models with feedback ==============================
def Feedbakk_FDD_model_scheduler_per_user(M, K, B, E, N_rf, k, more=1, qbit=0, output_all=False):
    inputs = Input((K, M))
    inputs_mod = tf.abs(inputs)
    encoding_module = CSI_reconstruction_model(M, K, B, E, N_rf, k, more=more)
    scheduling_module = FDD_per_user_architecture_return_all_softmaxes(M, K, k=k, N_rf=N_rf)
    # scheduling_module = FDD_per_user_architecture_double_softmax(M, K, k=k, N_rf=N_rf, output_all=output_all)
    reconstructed_input, z_qq, z_e = encoding_module(inputs_mod)
    scheduled_output, per_user_softmaxes, overall_softmax = scheduling_module(reconstructed_input)
    model = Model(inputs, [scheduled_output, z_qq, z_e, reconstructed_input, per_user_softmaxes, overall_softmax])
    return model
def Feedbakk_FDD_model_encoder_decoder(M, K, B, E, mul=1):
    inputs = Input((K, M))
    find_nearest_e = Closest_embedding_layer(user_count=K, embedding_count=2 ** B, bit_count=E, i=0)
    encoder = Autoencoder_Encoding_module((K, M), i=0, code_size=E, normalization=False)
    decoder = Autoencoder_Decoding_module(M, (K, E))
    z_e = encoder(inputs)
    z_qq = find_nearest_e(z_e)
    z_fed_forward = z_e + tf.stop_gradient(z_qq - z_e)
    out = decoder(z_fed_forward)
    output_all = tf.keras.layers.concatenate((out, z_qq, z_e), 2)
    # the output_all shape would look like
    model = Model(inputs, output_all, name="DiscreteVAE")
    return model
def Feedbakk_FDD_model_scheduler(M, K, B, E, N_rf, k, more=1, qbit=0, output_all=False, avg_max=None):
    inputs = Input((K, M))
    inputs_mod = tf.abs(inputs)
    encoding_module = CSI_reconstruction_model_seperate_decoders_input_mod(M, K, B, E, N_rf, k, more=more, qbit=qbit, avg_max=avg_max)
    scheduling_module = FDD_per_link_archetecture_more_G(M, K, k=k, N_rf=N_rf, normalization=False, avg_max=avg_max)
    # scheduling_module = FDD_per_user_architecture_double_softmax(M, K, k=k, N_rf=N_rf, output_all=output_all)
    reconstructed_input, z_qq, z_e = encoding_module(inputs_mod)
    # scheduled_output, raw_output = scheduling_module(reconstructed_input)
    scheduled_output, raw_output = scheduling_module(reconstructed_input)
    model = Model(inputs, [scheduled_output, raw_output, z_qq, z_e, reconstructed_input])
    print(model.summary())
    return model
def Feedbakk_FDD_model_scheduler_naive(M, K, B, E, N_rf, k, more=1, qbit=0, avg_max=None):
    inputs = Input((K, M))
    encoding_module = CSI_reconstruction_model_seperate_decoders_naive(M, K, B, E, N_rf, more=more, qbit=qbit, avg_max=avg_max)
    scheduling_module = FDD_per_link_archetecture_more_G(M, K, k, N_rf, normalization=False, avg_max=avg_max)
    # scheduling_module = FDD_per_user_architecture_double_softmax(M, K, k=k, N_rf=N_rf, output_all=output_all)
    reconstructed_input = encoding_module(inputs)
    scheduled_output, raw_output = scheduling_module(reconstructed_input)
    model = Model(inputs, [scheduled_output, raw_output, reconstructed_input])
    return model
def Feedbakk_FDD_model_scheduler_knowledge_distillation(M, K, B, E, N_rf, k, more=1, qbit=0, avg_max=None):
    inputs = Input((K, M))
    encoding_module = CSI_reconstruction_knowledge_distillation(M, K, B, E, N_rf, more=more, qbit=qbit, avg_max=avg_max)
    scheduling_module = FDD_per_link_archetecture_more_G(M, K, k, N_rf, normalization=False, avg_max=avg_max)
    # scheduling_module = FDD_per_user_architecture_double_softmax(M, K, k=k, N_rf=N_rf, output_all=output_all)
    reconstructed_input_teacher, reconstructed_input = encoding_module(inputs)

    scheduled_output, raw_output = scheduling_module(reconstructed_input)
    teacher_scheduled_output, teacher_raw_output = scheduling_module(reconstructed_input_teacher)
    model = Model(inputs, [scheduled_output, raw_output, teacher_scheduled_output, teacher_raw_output, reconstructed_input, reconstructed_input_teacher])
    return model
def CSI_reconstruction_model(M, K, B, E, N_rf, k, more=1):
    inputs = Input((K, M))
    inputs_mod = tf.abs(inputs)
    find_nearest_e = Closest_embedding_layer(user_count=K, embedding_count=2 ** B, bit_count=E, i=0)
    encoder = Autoencoder_Encoding_module((K, M), i=0, code_size=E * more, normalization=False)
    decoder = Autoencoder_Decoding_module(M * K, (K * E * more))
    z_e_all = encoder(inputs_mod)
    z_qq = find_nearest_e(z_e_all[:, :, :E])
    for i in range(1, more):
        z_qq = tf.concat((z_qq, find_nearest_e(z_e_all[:, :, E * i:E * (i + 1)])), axis=2)
    z_fed_forward = z_e_all + tf.stop_gradient(z_qq - z_e_all)
    z_fed_forward = tf.keras.layers.Reshape((K * E * more,))(z_fed_forward)
    reconstructed_input = tf.keras.layers.Reshape((K,M))(decoder(z_fed_forward))
    model = Model(inputs, [reconstructed_input, z_qq, z_e_all])
    return model
def CSI_reconstruction_knowledge_distillation(M, K, B, E, N_rf, more=1, qbit=0, avg_max=None):
    inputs = Input((K, M))
    inputs_mod = tf.abs(inputs)
    inputs_mod = tf.divide(inputs_mod, avg_max)
    encoder = Autoencoder_Encoding_module((K, M), i=0, code_size=more, normalization=False)
    decoder = Autoencoder_Decoding_module(M, (K, more), i = 0)
    decoder_teacher = Autoencoder_Decoding_module(M, (K, more), i = 1)
    z = encoder(inputs_mod)
    z_binary = tf.tanh(z) + tf.stop_gradient(tf.sign(z) - tf.tanh(z))
    reconstructed_input_teacher = tf.keras.layers.Reshape((K, M))(decoder_teacher(tf.tanh(z)))
    reconstructed_input = tf.keras.layers.Reshape((K, M))(decoder(z_binary))
    model = Model(inputs, [reconstructed_input_teacher, reconstructed_input])
    return model
def CSI_reconstruction_model_seperate_decoders(M, K, B, E, N_rf, k, more=1, qbit=0, avg_max=None):
    inputs = Input((K, M))
    inputs_mod = tf.abs(inputs)
    inputs_mod = tf.divide(inputs_mod, avg_max)
    find_nearest_e = Closest_embedding_layer(user_count=K, embedding_count=2 ** B, bit_count=E, i=0)
    encoder = Autoencoder_Encoding_module((K, M), i=0, code_size=E * more + qbit, normalization=False)
    decoder = Autoencoder_Decoding_module(M, (K, E * more))
    z_e_all = encoder(inputs_mod)
    z_e = z_e_all[:, :, :E * more]
    if qbit > 0:
        z_val = z_e_all[:, :, E * more:E * more + qbit]
        z_val = sigmoid(z_val) + tf.stop_gradient(binary_activation(z_val) - sigmoid(z_val)) + 0.1
    z_qq = find_nearest_e(z_e[:, :, :E])
    for i in range(1, more):
        z_qq = tf.concat((z_qq, find_nearest_e(z_e[:, :, E * i:E * (i + 1)])), axis=2)
    z_fed_forward = z_e + tf.stop_gradient(z_qq - z_e)
    if qbit > 0:
        z_fed_forward = tf.multiply(z_fed_forward, z_val)
    reconstructed_input = tf.keras.layers.Reshape((K, M))(decoder(z_fed_forward))
    model = Model(inputs, [reconstructed_input, z_qq, z_e])
    return model
def CSI_reconstruction_model_seperate_decoders_naive(M, K, B, E, N_rf, more=1, qbit=0, avg_max=None):
    inputs = Input((K, M))
    inputs_mod = tf.abs(inputs)
    inputs_mod = tf.divide(inputs_mod, avg_max)
    # inputs_mod = tf.keras.layers.Reshape((K, M, 1))(inputs_mod)
    # inputs_mod2 = tf.transpose(tf.keras.layers.Reshape((K, M, 1))(inputs_mod), perm=[0, 1, 3, 2])
    # inputs_mod = tf.keras.layers.Reshape((K, M * M))(tf.matmul(inputs_mod, inputs_mod2))
    # inputs_mod = tf.keras.layers.Reshape((K, M, 1))(inputs_mod)
    # inputs_mod2 = tf.transpose(tf.keras.layers.Reshape((K, M, 1))(inputs_mod), perm=[0, 1, 3, 2])
    # inputs_mod = tf.keras.layers.Reshape((K, M * M))(tf.matmul(inputs_mod, inputs_mod2))
    encoder = Autoencoder_Encoding_module((K, M), i=0, code_size=more, normalization=False)
    decoder = Autoencoder_Decoding_module(M, (K, more))
    z = encoder(inputs_mod)
    z = sigmoid(z) + tf.stop_gradient(binary_activation(sigmoid(z), shift=0.5) - sigmoid(z))
    reconstructed_input = tf.keras.layers.Reshape((K, M))(decoder(z))
    model = Model(inputs, reconstructed_input)
    return model
def CSI_reconstruction_model_seperate_decoders_chunky(M, K, B, E, N_rf, more=1, qbit=0, avg_max=None):
    inputs = Input((K, M))
    inputs_mod = tf.abs(inputs)
    inputs_mod = tf.divide(inputs_mod, avg_max)
    splits = 2
    code_size = more/splits
    encoder = Autoencoder_chunky_Encoding_module((K, M), i=0, code_size=code_size, splits=splits)
    decoder = Autoencoder_chunky_Decoding_module((K, int(more)), i=0, M=M, splits=splits)
    z = encoder(inputs_mod)
    z = sigmoid(z) + tf.stop_gradient(binary_activation(sigmoid(z), shift=0.5) - sigmoid(z))
    reconstructed_input = tf.keras.layers.Reshape((K, M))(decoder(z))
    model = Model(inputs, reconstructed_input)
    return model
def CSI_reconstruction_model_seperate_decoders_layers(M, K, B, E, N_rf, more=1, qbit=0, avg_max=None):
    inputs = Input((K, M))
    inputs_mod = tf.abs(inputs)
    inputs_mod = tf.divide(inputs_mod, avg_max)
    max_bits = 32
    splits = int(more/max_bits)
    if more <= max_bits:
        encoder = Autoencoder_Encoding_module((K, M), i=0, code_size=more, normalization=False)
        decoder = Autoencoder_Decoding_module(M, (K, more))
    else:
        encoder = Autoencoder_layers_Encoding_module((K, M), i=0, max_bits=max_bits, splits=splits)
        decoder = Autoencoder_layers_Decoding_module((K, int(more)), i=0, M=M, splits=splits)
    z = encoder(inputs_mod)
    z = sigmoid(z) + tf.stop_gradient(binary_activation(sigmoid(z), shift=0.5) - sigmoid(z))
    reconstructed_input = tf.keras.layers.Reshape((K, M))(decoder(z))
    model = Model(inputs, reconstructed_input)
    return model
def CSI_reconstruction_model_seperate_decoders_multirate(M, K, B, E, N_rf, rates, qbit=0, avg_max=None):
    # e.g. rates = [8,16,32,64]
    inputs = Input((K, M))
    inputs_mod = tf.abs(inputs)
    inputs_mod = tf.divide(inputs_mod, avg_max)
    splits = len(rates)
    more = tf.add_n(rates)
    encoder = Autoencoder_rates_Encoding_module((K, M), i=0, rates=rates, splits=splits)
    decoder = Autoencoder_rates_Decoding_module((K, int(more)), i=0, M=M, rates=rates, splits=splits)
    z = encoder(inputs_mod)
    z = sigmoid(z) + tf.stop_gradient(binary_activation(sigmoid(z), shift=0.5) - sigmoid(z))
    reconstructed_input = tf.keras.layers.Reshape((K, M))(decoder(z))
    model = Model(inputs, reconstructed_input)
    return model
def CSI_reconstruction_model_seperate_decoders_input_mod(M, K, B, E, N_rf, k, more=1, qbit=0, avg_max=None):
    inputs = Input((K, M))
    inputs_mod = tf.abs(inputs)
    # norm = tf.reduce_max(tf.keras.layers.Reshape((K * M,))(inputs_mod), axis=1, keepdims=True)
    # inputs_mod = tf.divide(inputs_mod, tf.expand_dims(norm, axis=1))
    inputs_mod = tf.divide(inputs_mod, avg_max)
    inputs_mod = tf.keras.layers.Reshape((K, M, 1))(inputs_mod)
    inputs_mod2 = tf.transpose(tf.keras.layers.Reshape((K, M, 1))(inputs_mod), perm=[0, 1, 3, 2])
    inputs_mod = tf.keras.layers.Reshape((K, M * M))(tf.matmul(inputs_mod, inputs_mod2))
    find_nearest_e = Closest_embedding_layer(user_count=K, embedding_count=B, bit_count=E, i=0)
    encoder = Autoencoder_Encoding_module((K, M * M), i=0, code_size=E * more + qbit, normalization=False)
    decoder = Autoencoder_Decoding_module(M, (K, E * more))
    z_e_all = encoder(inputs_mod)
    z_e = z_e_all[:, :, :E * more]
    if qbit > 0:
        z_val = z_e_all[:, :, E * more:E * more + qbit]
        z_val = sigmoid(z_val) + tf.stop_gradient(binary_activation(z_val) - sigmoid(z_val)) + 0.1
    z_qq = find_nearest_e(z_e[:, :, :E])
    for i in range(1, more):
        z_qq = tf.concat((z_qq, find_nearest_e(z_e[:, :, E * i:E * (i + 1)])), axis=2)
    z_fed_forward = z_e + tf.stop_gradient(z_qq - z_e)
    if qbit > 0:
        z_fed_forward = tf.multiply(z_fed_forward, z_val)
    reconstructed_input = tf.keras.layers.Reshape((K, M))(decoder(z_fed_forward))
    model = Model(inputs, [reconstructed_input, z_qq, z_e])
    return model
def CSI_reconstruction_model_seperate_decoders_DFT_matrix(M, K, B, E, N_rf, k, more=1, qbit=0):
    inputs = Input((K, M))
    inputs_mod = tf.abs(inputs)
    dft_matrix = tf.transpose(tf.abs(tf.constant(sp.linalg.dft(M), dtype=tf.complex64)), [1,0])
    norm = tf.reduce_max(tf.keras.layers.Reshape((K * M,))(inputs_mod), axis=1, keepdims=True)
    inputs_mod = tf.divide(inputs_mod, tf.expand_dims(norm, axis=1))
    # inputs_mod = tf.keras.layers.Reshape((K, M, 1))(inputs_mod)
    # inputs_mod2 = tf.transpose(tf.keras.layers.Reshape((K, M, 1))(inputs_mod), perm=[0, 1, 3, 2])
    # inputs_mod = tf.keras.layers.Reshape((K, M * M))(tf.matmul(inputs_mod, inputs_mod2))
    find_nearest_e = Closest_embedding_layer(user_count=K, embedding_count=2**1, bit_count=E, i=0)
    encoder = Autoencoder_Encoding_module((K, M), i=0, code_size=E * more + qbit, normalization=False)
    decoder = Autoencoder_Decoding_module(M, (K, E * more))
    z_e_all = encoder(inputs_mod)
    z_e = z_e_all[:, :, :E * more]
    if qbit > 0:
        z_val = z_e_all[:, :, E * more:E * more + qbit]
        z_val = sigmoid(z_val) + tf.stop_gradient(binary_activation(z_val) - sigmoid(z_val)) + 0.1
    z_qq = find_nearest_e(z_e[:, :, :E])
    for i in range(1, more):
        z_qq = tf.concat((z_qq, find_nearest_e(z_e[:, :, E * i:E * (i + 1)])), axis=2)
    z_fed_forward = z_e + tf.stop_gradient(z_qq - z_e)
    if qbit > 0:
        z_fed_forward = tf.multiply(z_fed_forward, z_val)
    reconstructed_input = tf.keras.layers.Reshape((K, M))(decoder(z_fed_forward))
    reconstructed_input = tf.matmul(dft_matrix, tf.transpose(reconstructed_input, perm=[0,2,1]))
    reconstructed_input = tf.transpose(reconstructed_input, perm=[0,2,1])
    model = Model(inputs, [reconstructed_input, z_qq, z_e])
    return model
def CSI_reconstruction_model_seperate_decoders_moving_avg_update(M, K, B, E, N_rf, k, more=1, qbit=0):
    inputs = Input((K, M))
    inputs_mod = tf.abs(inputs)
    find_nearest_e = Closest_embedding_layer_moving_avg(user_count=K, embedding_count=2 ** B, bit_count=E, i=0)
    encoder = Autoencoder_Encoding_module((K, M), i=0, code_size=E * more + qbit, normalization=False)
    decoder = Autoencoder_Decoding_module(M, (K, E * more))
    z_e_all = encoder(inputs_mod)
    z_e = z_e_all[:, :, :E * more]
    if qbit > 0:
        z_val = z_e_all[:, :, E * more:E * more + qbit]
        z_val = sigmoid(z_val) + tf.stop_gradient(binary_activation(z_val) - sigmoid(z_val)) + 0.1
    z_qq = find_nearest_e(z_e[:, :, :E])
    for i in range(1, more):
        z_qq = tf.concat((z_qq, find_nearest_e(z_e[:, :, E * i:E * (i + 1)])), axis=2)
    z_fed_forward = z_e + tf.stop_gradient(z_qq - z_e)
    if qbit > 0:
        z_fed_forward = tf.multiply(z_fed_forward, z_val)
    reconstructed_input = tf.keras.layers.Reshape((K, M))(decoder(z_fed_forward))
    model = Model(inputs, [reconstructed_input, z_qq, z_e])
    return model
def Floatbits_FDD_model_softmax(M, K, B):
    inputs = Input(shape=(K, M * 2 * 23), dtype=tf.float32)
    # create input vector
    reshaper = tf.keras.layers.Reshape((2 * M * K * 23,))
    x = reshaper(inputs)
    x = LeakyReLU()(x)
    x = Dense(M * K)(x)
    x = LeakyReLU()(x)
    x = Dense(M * M)(x)
    x = LeakyReLU()(x)
    x = Dense(M * K)(x)
    x_list2 = tf.split(x, num_or_size_splits=K, axis=1)
    output = tf.keras.layers.Softmax()(x_list2[0])
    for i in range(1, len(x_list2)):
        output = tf.concat((output, tf.keras.layers.Softmax()(x_list2[i])), axis=1)
    model = Model(inputs, output)
    return model
# def CSI_vanilla_reconstruction_model(M, K, B, E, N_rf, k, more=1):
def CSI_reconstruction_VQVAE2(M, K, B, E, N_rf, k, B_t=2, E_t=10, more=1):
    inputs = Input((K, M))
    inputs_mod = tf.abs(inputs)
    find_nearest_e_b = Closest_embedding_layer(user_count=K, embedding_count=2 ** B, bit_count=E, i=0)
    find_nearest_e_t = Closest_embedding_layer(user_count=K, embedding_count=2 ** B_t, bit_count=E_t, i=1)
    encoder_b = Autoencoder_Encoding_module((K, M + E_t), i=0, code_size=E * more, normalization=False)
    encoder_t = Autoencoder_Encoding_module((K, M), i=1, code_size=E_t * more, normalization=False)
    decoder_b = Autoencoder_Decoding_module(M, (K, (E_t + E) * more), i=0)
    # user side
    z_e_t = encoder_t(inputs_mod)
    z_q_t = find_nearest_e_t(z_e_t)
    z_fed_forward_t = z_e_t + tf.stop_gradient(z_q_t - z_e_t)
    z_e_b = encoder_b(tf.concat((z_fed_forward_t, inputs_mod), axis=2))
    z_q_b = find_nearest_e_b(z_e_b)
    z_fed_forward_b = z_e_b + tf.stop_gradient(z_q_b - z_e_b)
    z_in = tf.concat((z_fed_forward_t, z_fed_forward_b), axis=2)
    # base station side
    reconstructed_input = tf.keras.layers.Reshape((K, M))(decoder_b(z_in))
    model = Model(inputs, [reconstructed_input, z_q_b, z_e_b, z_q_t, z_e_t])
    print(model.summary())
    return model
def Feedbakk_FDD_model_scheduler_pretrained(M, K, B, E, N_rf, k, model_path, custome_obj, more=1, qbit=0, avg_max=None):
    inputs = Input((K, M))
    encoding_module = tf.keras.models.load_model(model_path, custom_objects=custome_obj)
    scheduling_module = FDD_per_link_archetecture_more_G(M, K, k, N_rf, normalization=False, avg_max=avg_max)
    # scheduling_module = FDD_per_user_architecture_double_softmax(M, K, k=k, N_rf=N_rf, output_all=output_all)
    reconstructed_input = encoding_module(inputs)
    scheduled_output, raw_output = scheduling_module(reconstructed_input)
    model = Model(inputs, [scheduled_output, raw_output, reconstructed_input])
    return model

if __name__ == "__main__":
    # F_create_encoding_model_with_annealing(2, 1, (2, 24))
    # F_create_CNN_encoding_model_with_annealing(2, 1, (2, 24))
    # print(Thresholdin_network((2, )).summary())
    # DiscreteVAE(2, 4, (2,))

    N = 1
    M = 64
    K = 50
    B = 3
    seed = 200
    N_rf = 4
    outputer = Stochastic_softmax_selectior_and_loss(M, K, N_rf)
    tim = tf.random.normal((10, M*K, N_rf), 0, 1)
    G = tf.random.normal((10, M*K), 0, 1)
    outputer(tim, tf.ones((10, M*K)), G, None)
    model = CSI_reconstruction_model_seperate_decoders_chunky
    FDD_RNN_model(M, K, N_rf)
    Top2Precoder_model(M, K, N_rf)
    G = generate_link_channel_data(N, K, M, N_rf)
    # mod = partial_feedback_top_N_rf_model(N_rf, B, 1, M, K, 0.1)
    # model = CSI_reconstruction_VQVAE2(M, K, B, 30, N_rf, 1, more=1)
    # model = DP_partial_feedback_pure_greedy_model(N_rf, B, 10, M, K, 1, perfect_CSI=True)
    # model(G)
    model = FDD_one_at_a_time(M, K, 6, N_rf)
    # LSTM_like_model_for_FDD(M, K, N_rf, k=3)
    # LSTM_like_model_for_FDD(M, K, k=3, N_rf=3)
    # nn = NN_Clustering(N_rf, M, reduced_dim=15)
    # nn.train_network(G)
    # nn.save_model("trained_models/Aug8th/nn_k_mean_dim_15/")
