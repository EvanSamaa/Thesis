from util import *
from models import *
import numpy as np
# from scipy.io import savemat
import tensorflow as tf
# from matplotlib import pyplot as plt

# = = = = = = = = = = = = = = = = = = on weighted Sumrates = = = = = = = = = = = = = = = = = = =
def test_BestWeight_weighted_SR(model, M = 20, K = 5, B = 10, N_rf = 5, sigma2_h = 6.3, sigma2_n = 0.00001, printing=True):
    store=np.zeros((8,))
    num_data = 20
    episodes = 200
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # tp_fn = ExpectedThroughput(name = "throughput")
    result = np.zeros((3,))
    loss_fn1 = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    loss_fn2 = Total_activation_limit_hard(K, M, N_rf=0)
    print("Testing Starts")
    tf.random.set_seed(200)
    np.random.seed(200)
    enviroment = Weighted_sumrate_model(K, M, N_rf, num_data, 0.05, True)
    # ds_load = generate_link_channel_data(num_data, K, M, 1)
    ds_load = gen_realistic_data("trained_models/Apr5th/one_hundred_user_config_0.npy", num_data, K, M, Nrf=N_rf)
    model = best_weight_model(N_rf, K, M)
    pred_tot = 0
    for e in range(0, episodes):
        enviroment.increment()
        pred = model(enviroment, ds_load)
        pred_tot = pred_tot + pred
        out = enviroment.compute_weighted_loss(pred, ds_load)
        result[0] = tf.reduce_mean(out)
        loss = tf.reduce_mean(loss_fn1(pred, ds_load))
        if printing:
            print("from the robst loss fn is ", loss)
            print("from the weighted sum rate fn is ", tf.reduce_mean(out))
            # print("the variance is ", tf.math.reduce_std(out))
        np.save("trained_models/Feb8th/user_loc0/best_weight/0p05/exp_avg_sumrate_best_weight_Nrf={}.npy".format(N_rf), enviroment.get_rates())
        np.save("trained_models/Feb8th/user_loc0/best_weight/0p05/weighted_sumrate_best_weight_Nrf={}.npy".format(N_rf),
                enviroment.get_weighted_rates())
    return store
def test_random_weighted_SR(model, M = 20, K = 5, B = 10, N_rf = 5, sigma2_h = 6.3, sigma2_n = 0.00001, printing=True):
    store=np.zeros((8,))
    num_data = 20
    episodes = 200
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # tp_fn = ExpectedThroughput(name = "throughput")
    result = np.zeros((3,))
    loss_fn1 = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    loss_fn2 = Total_activation_limit_hard(K, M, N_rf=0)
    print("Testing Starts")
    tf.random.set_seed(200)
    np.random.seed(200)
    enviroment = Weighted_sumrate_model(K, M, N_rf, num_data, 0.05, True)
    # ds_load = generate_link_channel_data(num_data, K, M, 1)
    ds_load = gen_realistic_data("trained_models/Apr5th/one_hundred_user_config_0.npy", num_data, K, M, Nrf=N_rf)
    model_rand_shape = [num_data, K*M, N_rf]
    for e in range(0, episodes):
        model_rand = np.random.uniform(0, 20, model_rand_shape)
        model_rand = np.array(model_rand, dtype=np.float32)
        model_rand = tf.nn.softmax(model_rand, axis=1)
        model_rand = tf.reduce_sum(model_rand, axis=2)
        pred = model_rand
        enviroment.increment()
        out = enviroment.compute_weighted_loss(pred, ds_load)
        result[0] = tf.reduce_mean(out)
        loss = tf.reduce_mean(loss_fn1(pred, ds_load))
        if printing:
            print("from the robst loss fn is ", loss)
            print("from the weighted sum rate fn is ", tf.reduce_mean(out))
    # enviroment.plot_activation(show=True)
        np.save("trained_models/Feb8th/user_loc0/random/0p05/exp_avg_sumrate_random_Nrf={}.npy".format(N_rf), enviroment.rates)
        np.save("trained_models/Feb8th/user_loc0/random/0p05/weighted_sumrate_random_Nrf={}.npy".format(N_rf),
                enviroment.weighted_rates)
    return store
def test_greedy_weighted_SR(model, M = 20, K = 5, B = 10, N_rf = 5, sigma2_h = 6.3, sigma2_n = 0.00001, printing=True):
    store=np.zeros((8,))
    num_data = 20
    episodes = 200
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # tp_fn = ExpectedThroughput(name = "throughput")
    result = np.zeros((3,))
    loss_fn1 = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    loss_fn2 = Total_activation_limit_hard(K, M, N_rf=0)
    print("Testing Starts")
    tf.random.set_seed(200)
    np.random.seed(200)
    enviroment = Weighted_sumrate_model(K, M, N_rf, num_data, 0.05, True)
    # ds_load = generate_link_channel_data(num_data, K, M, 1)
    # model = partial_feedback_pure_greedy_model_weighted_SR(N_rf, 0, 2, M, K, 1, enviroment)
    for e in range(0, episodes):
        ds_load = gen_realistic_data("trained_models/Apr5th/one_hundred_user_config_0.npy", num_data, K, M, Nrf=N_rf)
        enviroment.increment()
        pred = partial_feedback_pure_greedy_model_weighted_SR(N_rf, 0, 2, M, K, 1, enviroment)(ds_load)
        out = enviroment.compute_weighted_loss(pred, ds_load, update=True)
        result[0] = tf.reduce_mean(out)
        loss = tf.reduce_mean(loss_fn1(pred, ds_load))
        if printing:
            print("the soft result is ", result)
            print("from the robst loss fn is ", loss)
            # print("the variance is ", tf.math.reduce_std(out))
        np.save("trained_models/Feb8th/user_loc0/greedy/alpha0p01/exp_avg_sumrate_greedy_Nrf={}_0p05_alpha.npy".format(N_rf), enviroment.rates)
        np.save("trained_models/Feb8th/user_loc0/greedy/alpha0p01/weighted_sumrate_greedy_Nrf={}_0p05_alpha.npy".format(N_rf),
                enviroment.weighted_rates)
    return store
def test_PF_DFT_weighted_SR(model, M = 20, K = 5, B = 10, N_rf = 5, sigma2_h = 6.3, sigma2_n = 0.00001, printing=True, alpha = 0.1):
    store=np.zeros((8,))
    num_data = 20
    episodes = 200
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # tp_fn = ExpectedThroughput(name = "throughput")
    result = np.zeros((3,))
    loss_fn1 = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    loss_fn2 = Total_activation_limit_hard(K, M, N_rf=0)
    print("Testing Starts")
    tf.random.set_seed(200)
    np.random.seed(200)
    enviroment = Weighted_sumrate_model(K, M, N_rf, num_data, alpha, True)
    # ds_load = generate_link_channel_data(num_data, K, M, 1)
    # ds_load = gen_realistic_data("trained_models/Feb8th/user_loc0/one_hundred_user_config_0.npy", num_data, K, M, Nrf=N_rf)
    model = PF_DFT_model(M, K, N_rf, sigma2_n)
    for e in range(0, episodes):
        ds_load = gen_realistic_data("trained_models/Apr5th/one_hundred_user_config_0.npy", num_data, K, M,
                                     Nrf=N_rf)
        enviroment.increment()
        pred = model(ds_load, enviroment.get_weight())
        out = enviroment.compute_weighted_loss(pred, ds_load, update=True)
        result[0] = tf.reduce_mean(out)
        loss = tf.reduce_mean(loss_fn1(pred, ds_load))
        if printing:
            print("the soft result is ", result)
            print("from the robst loss fn is ", loss)
            # print("the variance is ", tf.math.reduce_std(out))
    # enviroment.plot_activation(show=True)
        np.save("trained_models/Feb8th/user_loc0/PFDFT/Test_different_alphas/exp_avg_sumrate_PFDFT_Nrf={}_0p0{}_alpha.npy".format(N_rf, int(alpha*100)), enviroment.rates)
        np.save("trained_models/Feb8th/user_loc0/PFDFT/Test_different_alphas/weighted_sumrate_PFDFT_Nrf={}_0p0{}_alpha.npy".format(N_rf, int(alpha*100)),
                enviroment.weighted_rates)
    return store
def test_performance_weighted_SR(model, M=20, K=5, B=10, N_rf=5, sigma2_h=6.3, sigma2_n=0.00001):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # tp_fn = ExpectedThroughput(name = "throughput")
    num_data = 20
    num_episodes = 200
    tf.random.set_seed(200)
    np.random.seed(200)
    result = np.zeros((3,))
    test_env = Weighted_sumrate_model(K, M, N_rf, num_data, .05, hard_decision=True)
    # loss_fn1 = tf.keras.losses.MeanSquaredError()
    # loss_fn1 = Sum_rate_utility_RANKING_hard(K, M, sigma2_n, N_rf, True)
    # loss_fn2 = Bin arization_regularization(K, num_data, M, k=N_rf)
    loss_fn2 = Total_activation_limit_hard(K, M, N_rf=0)
    print("Testing Starts")
    ds = gen_realistic_data("trained_models/Apr5th/one_hundred_user_config_0.npy", num_data, K, M, N_rf)
    for e in range(0, num_episodes):
        test_env.increment()
        # ds = generate_link_channel_data(num_data, K, M, N_rf)
        # ds, angle = generate_link_channel_data_with_angle(num_data, K, M)
        # print(ds)
        ds_load = ds
        # if e > 0:
        #     ds_load = ds * tf.complex(tf.expand_dims(test_env.get_binary_weights(), axis=2), 0.0)
        input_mod = tf.concat([ds_load, tf.complex(tf.expand_dims(test_env.get_binary_weights(), axis=2), 0.0)],
                              axis=2)
        scheduled_output, raw_output = model.predict(input_mod, batch_size=50)
        prediction = scheduled_output[:, -1]
        prediction_hard = Harden_scheduling_user_constrained(N_rf, K, M)(prediction)
        out_hard = test_env.compute_weighted_loss(prediction, ds_load, update=True)
        sr = test_env.compute_raw_loss(prediction, ds_load)
        result[0] = tf.reduce_mean(out_hard)
        result_2 = tf.reduce_mean(tf.reduce_sum(test_env.get_rates(), axis=2))
        result[1] = loss_fn2(prediction_hard)
        print("the top Nrf result for time: {} is ".format(e), result[0], result_2, "and without the weight it is ", tf.reduce_mean(sr))
    # test_env.plot_average_rates(True)
    data = test_env.rates
    np.save(
        "trained_models/Feb8th/user_loc0/train_with_0_1_weight/exp_avg_sumrate_dnn_Nrf={}_0p05_alpha.npy".format(N_rf),
        test_env.rates)
    np.save(
        "trained_models/Feb8th/user_loc0/train_with_0_1_weight/weighted_sumrate_dnn_Nrf={}_0p05_alpha.npy".format(N_rf),
        test_env.weighted_rates)

    # test_env.plot_activation(True)
        # ========= ========= =========  plotting ========= ========= =========
        # ds = tf.square(tf.abs(ds))
        # # prediction = prediction[:, :, 2]
        # unflattened_X = tf.reshape(prediction, (prediction.shape[0], K, M))
        # unflattened_X = tf.transpose(unflattened_X, perm=[0, 2, 1])
        # denominator = tf.matmul(ds, unflattened_X)
        # for i in range(0, num_data):
        #     plt.imshow(denominator[i], cmap="gray")
        #     plt.show(block=False)
        #     plt.pause(0.0001)
        #     plt.close()
        # ========= ========= =========  plotting ========= ========= =========

def greedy_grid_search():
    M = 64
    K = 50
    sigma2_n = 1
    N_rf = 8
    sigma2_h = 0.0001
    out = np.zeros((64, 32, 8))
    bits_to_try = [1, 2, 3, 4, 5, 6, 7] + list(range(8, 33, 4))
    for links in range(1, 19):
        for bits in bits_to_try:
            if links * (6 + bits) <= 128:
                model = DP_partial_feedback_pure_greedy_model(8, bits, links, M, K, sigma2_n, perfect_CSI=False)
                losses = test_greedy(model, M=M, K=K, B=bits, N_rf=N_rf, sigma2_n=sigma2_n, sigma2_h=sigma2_h)
                out[links-1, bits-1, :] = losses
                np.save("trained_models/Dec_13/greedy_save_here/grid_search_all_under128_30AOE_min_max_quantization.npy", out)
                print("{} links {} bits is done".format(links, bits))
def partial_feedback_and_DNN_grid_search():
    M = 64
    K = 50
    sigma2_n = 1
    N_rf = 8
    sigma2_h = 0.0001
    model_path = "trained_models/Jan_18/test_dnn_Nrf={}100xmutex.h5"
    bits_to_try = [1,2,3,4,5,6,7] + list(range(8, 32, 4))
    out = np.zeros((64, 32, 8))
    for links in range(1, 19):
        for bits in bits_to_try:
            if links*(6+bits) <= 128:
                # garbage, g_max = Input_normalization_per_user(
                #     tf.abs(generate_link_channel_data_fullAOE(1000, K, M, Nrf=1)))
                # feed_back_model = k_link_feedback_model(N_rf, bits, links, M, K, g_max)
                feed_back_model = max_min_k_link_feedback_model(1, bits, links, M, K)
                losses = test_performance_partial_feedback_and_DNN_all_Nrf(feed_back_model, model_path, M=M, K=K, B=bits,
                                                          sigma2_n=sigma2_n, sigma2_h=sigma2_h)
                out[links-1, bits-1] = np.maximum(out[links-1, bits-1], -losses)
                np.save("trained_models/Dec_13/greedy_save_here/trainedOn180_runOn30.npy", out)
                print("{} links {} bits is done".format(links, bits))
def test_greedy(model, M = 20, K = 5, B = 10, N_rf = 5, sigma2_h = 6.3, sigma2_n = 1, printing=True):
    store=np.zeros((8,))
    num_data = 1
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # tp_fn = ExpectedThroughput(name = "throughput")
    result = np.zeros((3,))
    loss_fn1 = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    loss_fn2 = Total_activation_limit_hard(K, M, N_rf=0)
    print("Testing Starts")
    tf.random.set_seed(200)
    np.random.seed(200)
    # ds_load = generate_link_channel_data(num_data, K, M, 1)
    ds_load = gen_realistic_data("trained_models/Apr5th/one_hundred_user_config_0.npy", num_data, K, M, Nrf=1)
    # print(ds_load.shape)
    prediction = model(ds_load)
    counter = 1
    for i in prediction:
        i_complex = tf.complex(tf.sqrt(counter*1.0), 0.0)
        out = loss_fn1(i, tf.abs(ds_load/i_complex))
        result[0] = tf.reduce_mean(out)
        result[1] = loss_fn2(i)
        if printing:
            print("the soft result is ", result)
            print("the variance is ", tf.math.reduce_std(out))
        store[counter-1] = tf.reduce_mean(out)
        counter = counter + 1
    return store
def test_DNN_greedy(model, M = 20, K = 5, B = 10, N_rf = 5, sigma2_h = 6.3, sigma2_n = 0.00001, printing=True):
    store=np.zeros((8,))
    num_data = 5
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # tp_fn = ExpectedThroughput(name = "throughput")
    result = np.zeros((3,))
    loss_fn1 = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    loss_fn2 = Total_activation_limit_hard(K, M, N_rf=0)
    print("Testing Starts")
    tf.random.set_seed(200)
    np.random.seed(200)
    ds_load = generate_link_channel_data(num_data, K, M, 1)
    prediction = model(ds_load)
    counter = 1
    for i in prediction:
        i_complex = tf.complex(tf.sqrt(counter*1.0), 0.0)
        out = loss_fn1(i, tf.abs(ds_load/i_complex))
        result[0] = tf.reduce_mean(out)
        result[1] = loss_fn2(i)
        if printing:
            print("the soft result is ", result)
            print("the variance is ", tf.math.reduce_std(out))
        store[counter-1] = tf.reduce_mean(out)
        counter = counter + 1
    return store
def test_greedy_different_resolution(M = 20, K = 5, B = 10, N_rf = 5, sigma2_h = 6.3, sigma2_n = 0.00001):
    num_data = 1000
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    # tp_fn = ExpectedThroughput(name = "throughput")
    result = np.zeros((64,))
    print("Testing Starts")
    ds = generate_link_channel_data(num_data, K, M, Nrf=N_rf)
    for i in range(1, 64+1):
        loss_fn1 = Sum_rate_utility_WeiCui(K, M, sigma2_n)
        model = partial_feedback_pure_greedy_model_not_perfect_CSI_available(N_rf, 32, i, M, K, sigma2_n)
        output = model(ds)
        out = loss_fn1(output, tf.abs(ds))
        result[i-1] = tf.reduce_mean(out)
        print("the result for {} is ".format(i), out)
        np.save("trained_models/Sept14th/greedy_resolution_change.npy", result)
def test_DNN_different_K(file_name, M = 20, K = 5, B = 10, N_rf = 5, sigma2_h = 6.3, sigma2_n = 0.00001):
    num_data = 10
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # tp_fn = ExpectedThroughput(name = "throughput")
    result = np.zeros((3,))
    print("Testing Starts")
    ds = generate_link_channel_data(num_data, 60, M)
    Ks = [10, 20, 30, 40, 50, 60]
    for i in Ks:
        ds_load = ds[:, :i]
        loss_fn1 = Sum_rate_utility_WeiCui(i, M, sigma2_n)
        loss_fn2 = Total_activation_limit_hard(i, M, N_rf=0)
        model = tf.keras.models.load_model(file_name.format(i), custom_objects=custome_obj)
        prediction = model.predict(ds_load)[0][:, -1]
        out = loss_fn1(prediction, tf.abs(ds_load))
        result[0] = tf.reduce_mean(out)
        result[1] = loss_fn2(prediction)
        print("the soft result is ", result)
        print("the variance is ", tf.math.reduce_std(out))

        prediction_binary = binary_activation(prediction)
        out_binary = loss_fn1(prediction_binary, ds_load)
        result[0] = tf.reduce_mean(out_binary)
        result[1] = loss_fn2(prediction_binary)
        print("the hard result is ", result)
        print("the variance for binary result is ", tf.math.reduce_std(out_binary))

        prediction_hard = Harden_scheduling(k=N_rf)(prediction)
        out_hard = loss_fn1(prediction_hard, ds_load)
        result[0] = tf.reduce_mean(out_hard)
        result[1] = loss_fn2(prediction_hard)
        print("the top Nrf result is ", result)
        print("the variance for hard result is ", tf.math.reduce_std(out_hard))
def different_greedy(m1, m2, M = 20, K = 5, B = 10, N_rf = 5, sigma2_h = 6.3, sigma2_n = 0.00001):
    num_data = 100
    loss_fn1 = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    ds = generate_link_channel_data(num_data, K, M)
    max_difference = 0
    worstG = None
    for i in range(num_data):
        gain_1 = loss_fn1(m1(ds[i:i+1]), ds[i:i+1])
        gain_2 = loss_fn1(m2(ds[i:i+1]), ds[i:i+1])
        difference = tf.abs(tf.abs(gain_1) - tf.abs(gain_2))
        if difference >= max_difference:
            max_difference = difference
            worstG = ds[i]
        if difference > 11:
            break
    print(worstG)
    np.save("G_with_most_difference.npy", np.array(worstG))
    # savemat("G_with_most_difference.mat", {"G":np.array(worstG)})
def test_performance(model, M = 20, K = 5, B = 10, N_rf = 5, sigma2_h = 6.3, sigma2_n = 0.00001):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # tp_fn = ExpectedThroughput(name = "throughput")
    num_data = 20
    result = np.zeros((3, ))
    loss_fn1 = Sum_rate_utility_WeiCui(K, M, sigma2_n)

    # loss_fn1 = tf.keras.losses.MeanSquaredError()
    # loss_fn1 = Sum_rate_utility_RANKING_hard(K, M, sigma2_n, N_rf, True)
    # loss_fn2 = Bin arization_regularization(K, num_data, M, k=N_rf)
    loss_fn2 = Total_activation_limit_hard(K, M, N_rf = 0)
    print("Testing Starts")
    for e in range(0, 1):
        # ds = generate_link_channel_data(num_data, K, M, N_rf)
        ds = gen_realistic_data("trained_models/Apr5th/one_hundred_user_config_0.npy", num_data, K, M, N_rf)
        # ds, angle = generate_link_channel_data_with_angle(num_data, K, M)
        # print(ds)
        ds_load = ds
        scheduled_output, raw_output = model.predict(ds, batch_size=50)
        prediction = scheduled_output[:, -1]
        # prediction = model(ds_load)[-1]
        # scheduled_output, raw_output, input_mod, input_reconstructed_mod, reconstructed_input = model.predict_on_batch(ds_load)

        # scheduled_output, raw_output, recon = model(ds_load)

        # for i in range(0, num_data):
        # from matplotlib import pyplot as plt
        # for k in range(4, 5):
        #     G_pred = DP_partial_feedback_pure_greedy_model(N_rf, 64, 10, M, K, sigma2_n, True)(ds_load[k:k+1])
        #     for i in range(0,5):
        #         prediction = scheduled_output[:, i]
        #         prediction_hard = Harden_scheduling_user_constrained(N_rf, K, M)(prediction)
        #         # plt.imshow(tf.reshape(prediction[k], (K, M)))
        #         # plt.plot(np.arange(0, K*M), G_pred[-1][0])
        #         plt.plot(np.arange(0, K*M), prediction_hard[k])
        #         plt.plot(np.arange(0, K*M), prediction[k])
        #
        #         plt.show()
        #     # tf.concat([reconstructed_input[k], tf.zeros((50, 20)), tf.abs(ds_load[k])], axis=1)
        #
        #     plt.show()
        # prediction = scheduled_output[:, -1]
        out = loss_fn1(prediction, tf.abs(ds_load))
        result[0] = tf.reduce_mean(out)
        result[1] = loss_fn2(prediction)
        print("the soft result is ", result)
        print("the variance is ", tf.math.reduce_std(out))

        prediction_binary = binary_activation(prediction)
        out_binary = loss_fn1(prediction_binary, ds_load)
        result[0] = tf.reduce_mean(out_binary)
        result[1] = loss_fn2(prediction_binary)
        print("the hard result is ", result)
        print("the variance for binary result is ", tf.math.reduce_std(out_binary))

        prediction_hard = Harden_scheduling_user_constrained(N_rf, K, M)(prediction)
        out_hard = loss_fn1(prediction_hard, ds_load)
        result[0] = tf.reduce_mean(out_hard)
        result[1] = loss_fn2(prediction_hard)
        print("the top Nrf result is ", result)
        print("the variance for hard result is ", tf.math.reduce_std(out_hard))
        # ========= ========= =========  plotting ========= ========= =========
        # ds = tf.square(tf.abs(ds))
        # # prediction = prediction[:, :, 2]
        # unflattened_X = tf.reshape(prediction, (prediction.shape[0], K, M))
        # unflattened_X = tf.transpose(unflattened_X, perm=[0, 2, 1])
        # denominator = tf.matmul(ds, unflattened_X)
        # for i in range(0, num_data):
        #     plt.imshow(denominator[i], cmap="gray")
        #     plt.show(block=False)
        #     plt.pause(0.0001)
        #     plt.close()
        # ========= ========= =========  plotting ========= ========= =========
def test_performance_partial_feedback_and_DNN(feed_back_model, dnn_model, M = 20, K = 5, B = 10, N_rf = 5, sigma2_h = 6.3, sigma2_n = 0.00001):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # tp_fn = ExpectedThroughput(name = "throughput")
    num_data = 500
    result = np.zeros((3, ))
    loss_fn1 = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    # loss_fn1 = tf.keras.losses.MeanSquaredError()
    # loss_fn1 = Sum_rate_utility_RANKING_hard(K, M, sigma2_n, N_rf, True)
    # loss_fn2 = Bin arization_regularization(K, num_data, M, k=N_rf)
    loss_fn2 = Total_activation_limit_hard(K, M, N_rf = 0)
    # print("Testing Starts")
    rtv = 0
    for e in range(0, 1):
        ds = generate_link_channel_data(num_data, K, M, N_rf)
        # ds, angle = generate_link_channel_data_with_angle(num_data, K, M)
        # print(ds)
        ds_load = ds
        ds_load_q = feed_back_model(ds_load)
        from matplotlib import pyplot as plt
        plt.imshow(np.abs(ds_load_q[0]))
        plt.show()
        scheduled_output, raw_output = dnn_model.predict(ds_load_q, batch_size=50)
        prediction = scheduled_output[:, -1]
        out = loss_fn1(prediction, tf.abs(ds_load))
        result[0] = tf.reduce_mean(out)
        result[1] = loss_fn2(prediction)
        print("the soft result is ", result)
        print("the variance is ", tf.math.reduce_std(out))

        prediction_binary = binary_activation(prediction)
        out_binary = loss_fn1(prediction_binary, ds_load)
        result[0] = tf.reduce_mean(out_binary)
        result[1] = loss_fn2(prediction_binary)
        print("the hard result is ", result)
        print("the variance for binary result is ", tf.math.reduce_std(out_binary))

        prediction_hard = Harden_scheduling_user_constrained(N_rf, K, M)(prediction)
        out_hard = loss_fn1(prediction_hard, ds_load)
        result[0] = tf.reduce_mean(out_hard)
        result[1] = loss_fn2(prediction_hard)
        print("the top Nrf result is ", result)
        print("the variance for hard result is ", tf.math.reduce_std(out_hard))
        rtv = result[0]
        # ========= ========= =========  plotting ========= ========= =========
        # ds = tf.square(tf.abs(ds))
        # # prediction = prediction[:, :, 2]
        # unflattened_X = tf.reshape(prediction, (prediction.shape[0], K, M))
        # unflattened_X = tf.transpose(unflattened_X, perm=[0, 2, 1])
        # denominator = tf.matmul(ds, unflattened_X)
        # for i in range(0, num_data):
        #     plt.imshow(denominator[i], cmap="gray")
        #     plt.show(block=False)
        #     plt.pause(0.0001)
        #     plt.close()
        # ========= ========= =========  plotting ========= ========= =========
        return rtv
def test_performance_partial_feedback_and_DNN_all_Nrf(feed_back_model, dnn_model_path, M = 20, K = 5, B = 10, sigma2_h = 6.3, sigma2_n = 0.00001):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # tp_fn = ExpectedThroughput(name = "throughput")
    num_data = 500
    result = np.zeros((3, ))
    loss_fn1 = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    loss_fn2 = Total_activation_limit_hard(K, M, N_rf = 0)
    print("Testing Starts")
    rtv = np.zeros((8,))
    for e in range(0, 1):
        tf.random.set_seed(200)
        np.random.seed(200)
        ds = generate_link_channel_data(num_data, K, M, 1)
        ds_load = ds
        ds_load_q_origial = feed_back_model(ds_load)
        for N_rf in range(1, 9):
            ds_load_q = ds_load_q_origial/tf.sqrt(N_rf * 1.0)
            dnn_model = tf.keras.models.load_model(dnn_model_path.format(N_rf), custom_objects=custome_obj)
            scheduled_output, raw_output = dnn_model.predict(ds_load_q, batch_size=50)
            prediction = scheduled_output[:, -1]
            out = loss_fn1(prediction, tf.abs(ds_load)/tf.sqrt(N_rf * 1.0))
            result[0] = tf.reduce_mean(out)
            result[1] = loss_fn2(prediction)
            # print("the soft result is ", result)
            # print("the variance is ", tf.math.reduce_std(out))

            prediction_binary = binary_activation(prediction)
            out_binary = loss_fn1(prediction_binary, tf.abs(ds_load)/tf.sqrt(N_rf * 1.0))
            result[0] = tf.reduce_mean(out_binary)
            result[1] = loss_fn2(prediction_binary)
            # print("the hard result is ", result)
            # print("the variance for binary result is ", tf.math.reduce_std(out_binary))

            prediction_hard = Harden_scheduling_user_constrained(N_rf, K, M)(prediction)
            out_hard = loss_fn1(prediction_hard, tf.abs(ds_load)/tf.sqrt(N_rf * 1.0))
            result[0] = tf.reduce_mean(out_hard)
            result[1] = loss_fn2(prediction_hard)
            # print("the top Nrf result is ", result)
            # print("the variance for hard result is ", tf.math.reduce_std(out_hard))
            rtv[N_rf-1] = result[0]
        print(rtv)
            # ========= ========= =========  plotting ========= ========= =========
            # ds = tf.square(tf.abs(ds))
            # # prediction = prediction[:, :, 2]
            # unflattened_X = tf.reshape(prediction, (prediction.shape[0], K, M))
            # unflattened_X = tf.transpose(unflattened_X, perm=[0, 2, 1])
            # denominator = tf.matmul(ds, unflattened_X)
            # for i in range(0, num_data):
            #     plt.imshow(denominator[i], cmap="gray")
            #     plt.show(block=False)
            #     plt.pause(0.0001)
            #     plt.close()
            # ========= ========= =========  plotting ========= ========= =========
        tf.keras.backend.clear_session()
        return rtv
def plot_data(arr, col=[], title="loss", series_name = None):
    # from matplotlib import pyplot as plt
    cut = 0
    for i in range(arr.shape[0]-1, 0, -1):
        for k in range(0, arr.shape[1]):
            if arr[i,k] != 0:
                cut = i
                break
    arr = arr[:cut, :]

    x = np.arange(0, arr.shape[0])
    for i in range(len(col)):
        plt.plot(x, arr[:, col[i]], '+', label=series_name[i])
    plt.legend()
    # plt.plot(x, arr[:, 3])
    plt.xlabel("epochs")
    plt.ylabel("sum rate")
    plt.title(title)
    # plt.show()
def garsons_method(model_path):
    from matplotlib import pyplot as plt
    model = tf.keras.models.load_model(model_path, custom_objects=custome_obj)
    dnn = model.get_layer("DNN_within_model0")
    weights = dnn.get_layer("Dense1_inside_DNN0")
    kernel = weights.kernel
    garson_importance = tf.reduce_sum(tf.abs(kernel), axis=1)
    norm = tf.reduce_sum(garson_importance, keepdims=True)
    garson_importance = tf.divide(garson_importance, norm)
    plt.plot(garson_importance.numpy())
    plt.show()
def all_bits_compare_with_greedy():
    file1 = "trained_models/Dec28/NRF=5/pretrained_encoder/GNN_annealing_temp_B={}+limit_res=6.npy"
    file2 = "trained_models/Dec28/NRF=5/GNN_annealing_temp_B={}+limit_res=6.npy"
    y = []
    x = []
    # for i in range(1,110,2):
    for i in range(2, 120, 2):
        out = np.load(file1.format(i))
        check = 1
        while True:
            if out[check,-1] != 0:
                break
            check += 1
        print(i, -out[:,-1].min())
        x.append(i)
        y.append(-out[:,-1].min())
    from matplotlib import pyplot as plt
    plt.plot(np.array(x), np.array(y), label = "pretrained encoder")

    y = []
    x = []
    for i in range(1, 100, 2):
        try:
            out = np.load(file2.format(i))
        except:
            out = np.load("trained_models/Dec28/NRF=5/GNN_annealing_temp_B={}+limit_res=6.h5.npy".format(i))
        check = 1
        while True:
            if out[check,-1] != 0:
                break
            check += 1
        print(i, -out[:,-1].min())
        x.append(i)
        y.append(-out[:,-1].min())
    from matplotlib import pyplot as plt
    plt.plot(np.array(x), np.array(y), label = "STE")

    grid = np.load("trained_models/Dec_13/greedy_save_here/grid_search_all_under128.npy")
    # add or remove points using x and y
    x = np.arange(1, 64)  # links
    y = np.arange(1, 32)  # bits
    Nrf = 5
    out = np.zeros((128,))
    out_x = []
    out_y = []
    for i in x:
        for j in y:
            if grid[i, j].any() != 0:
                out[i * (6 + j)] = max(-grid[i - 1, j - 1, Nrf - 1], out[i * (6 + j)])
    for i in range(0, 128):
        if out[i] != 0:
            out_x.append(i)
            out_y.append(out[i])

    plt.plot(np.array(out_x), np.array(out_y), label="greedy".format(i))
    plt.xlabel("bits per user")
    plt.ylabel("sum rate")
    plt.legend()
    plt.show()
def compare_quantizers(p=1):
    N = 1000
    G = tf.abs(generate_link_channel_data(N, K, M, Nrf=1))
    top_values, top_indices = tf.math.top_k(G, k=p)
    G_copy = np.zeros((top_indices.shape[0], K, M))
    for n in range(0, top_indices.shape[0]):
        for i in range(0, K * p):
            p_i = int(i % p)
            user_i = int(tf.floor(i / p))
            G_copy[n, user_i, int(top_indices[n, user_i, p_i])] = top_values[n, user_i, p_i]
    G_copy = tf.constant(G_copy, dtype=tf.float32)
    G_top_p = G_copy
    def q_1(G_topk, B): # mean of row max
        g_max = tf.reduce_max(G_topk, axis=2)
        g_max = tf.reduce_mean(g_max)
        G_temp = tf.divide(G_topk, g_max)
        G_temp = tf.where(G_temp > 1, 1, G_temp)
        G_temp = tf.round(G_temp * (2 ** B - 1)) / (2 ** B - 1)
        G_temp = tf.multiply(G_temp, g_max)
        return G_temp
    def q_2(G_topk, B): # mean of matrix max
        g_max = tf.reshape(G_topk, (G_topk.shape[0], M*K))
        g_max = tf.reduce_max(g_max, axis = 1)
        g_max = tf.reduce_mean(g_max)
        G_temp = tf.divide(G_topk, g_max)
        G_temp = tf.where(G_temp > 1, 1, G_temp)
        G_temp = tf.round(G_temp * (2 ** B - 1)) / (2 ** B - 1)
        G_temp = tf.multiply(G_temp, g_max)
        return G_temp
    def q_3(G_topk, B): # matrix max
        g_max = tf.reduce_max(G_topk)
        G_temp = tf.divide(G_topk, g_max)
        G_temp = tf.where(G_temp > 1, 1, G_temp)
        G_temp = tf.round(G_temp * (2 ** B - 1)) / (2 ** B - 1)
        G_temp = tf.multiply(G_temp, g_max)
        return G_temp
    def q_4(G_topk, B): # limited range between max and min
        g_max = tf.reshape(G_topk, (G_topk.shape[0], M * K))
        g_max = tf.reduce_max(g_max, axis=1)
        g_max = tf.reduce_mean(g_max)
        g_min = tf.reshape(G_topk, (N, M*K))
        g_min = tf.reduce_mean(tf.sort(g_min, axis=1)[:, (M-p)*K])
        mask = tf.where(G_topk != 0, 1.0, 0.0)
        G_temp = tf.divide(G_topk-mask*g_min, (g_max-g_min))
        G_temp = tf.where(G_temp > 1, 1, G_temp)
        G_temp = tf.round(G_temp * (2 ** B - 1)) / (2 ** B - 1)
        G_temp = tf.multiply(G_temp, (g_max-g_min)) + g_min * mask
        return G_temp

    quantizers = [q_1, q_2, q_3, q_4]
    output = np.zeros((16, 4))
    for quantizer in range(4):
        outputs = []
        for bits in range(1, 17):
            g_i = quantizers[quantizer](G_top_p, bits)
            error = tf.abs(G_copy - g_i)
            error = tf.reduce_mean(error).numpy()
            outputs.append(error)
        output[:, quantizer] = np.array(outputs)
    np.save("trained_models/quantization_comparisons/link={}_basic4.npy".format(p),output)
def all_bits_compare_with_greedy_plot_link_seperately():
    file1 = "trained_models/Dec28/NRF=5/pretrained_encoder/GNN_annealing_temp_B={}+limit_res=6.npy"
    file2 = "trained_models/Dec28/NRF=5/GNN_annealing_temp_B={}+limit_res=6.npy"
    y = []
    x = []
    # for i in range(1,110,2):
    for i in range(2, 120, 2):
        out = np.load(file1.format(i))
        check = 1
        while True:
            if out[check,-1] != 0:
                break
            check += 1
        print(i, -out[:,-1].min())
        x.append(i)
        y.append(-out[:,-1].min())
    from matplotlib import pyplot as plt
    plt.plot(np.array(x), np.array(y), label = "pretrained encoder")
    grid = np.load("trained_models/Dec_13/greedy_save_here/grid_search_all_under128.npy")
    # add or remove points using x and y
    x = np.arange(1, 64)  # links
    y = np.arange(1, 32)  # bits
    Nrf = 5
    out = np.zeros((128,))
    out_x = []
    out_y = []
    for i in x:
        for j in y:
            if grid[i, j].any() != 0:
                out[i * (6 + j)] = max(-grid[i - 1, j - 1, Nrf - 1], out[i * (6 + j)])
    for i in range(0, 128):
        if out[i] != 0:
            out_x.append(i)
            out_y.append(out[i])

    plt.plot(np.array(out_x), np.array(out_y), label="greedy".format(i))
    plt.xlabel("bits per user")
    plt.ylabel("sum rate")
    plt.legend()
    plt.show()
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    Nrf = 8
    file_tp = "trained_models/Feb8th/user_loc0/PFDFT/Test_different_alphas/exp_avg_sumrate_PFDFT_Nrf={}_0p05_alpha.npy".format(Nrf)
    file_tp2 = "trained_models/Feb8th/user_loc0/train_with_0_1_weight/exp_avg_sumrate_dnn_Nrf={}_0p05_alpha.npy".format(Nrf)
    file_tp3 = "trained_models/Feb8th/user_loc0/greedy/alpha0p01/exp_avg_sumrate_greedy_Nrf={}_0p05_alpha.npy".format(Nrf)
    fn = file_tp
    a0p01 = np.load(fn).sum(axis=1).mean(axis=0)
    plt.plot(-np.sort(-a0p01), label = "PFDFT")
    a0p01 = np.load(file_tp2).sum(axis=1).mean(axis=0)
    plt.plot(-np.sort(-a0p01), label="dnn")
    a0p01 = np.load(file_tp3).sum(axis=1).mean(axis=0)
    plt.plot(-np.sort(-a0p01), label="greedy")
    plt.legend()
    plt.show()
    A[2]
    # file1 = "trained_models/Feb8th/user_loc0/PFDFT/0p05/exp_avg_sumrate_PFDFT_Nrf={}_0p01_alpha.npy".format(Nrf)
    # file2 = "trained_models/Feb8th/user_loc0/random/0p05/exp_avg_sumrate_random_Nrf={}.npy".format(Nrf)
    # file3 = "trained_models/Feb8th/user_loc0/best_weight/0p05/exp_avg_sumrate_best_weight_Nrf={}.npy".format(Nrf)
    # # file1 = "trained_models/Feb8th/user_loc0/PFDFT/weighted_sumrate_PFDFT_Nrf=8_0p01_alpha.npy"
    # # file2 = "trained_models/Feb8th/user_loc0/random/weighted_sumrate_random_Nrf=8.npy"
    # # file3 = "trained_models/Feb8th/user_loc0/best_weight/alpha=0p01/weighted_sumrate_best_weight_Nrf=8.npy"
    # a0p01 = np.load(file1).sum(axis=1).mean(axis=0)
    # random = np.load(file2).sum(axis=1).mean(axis=0)
    # best_weight = np.load(file3).sum(axis=1).mean(axis=0)
    # # plt.subplot(1, 2, 1)
    # plt.plot(-np.sort(-a0p01))
    # plt.plot(random)
    # plt.plot(-best_weight)
    # plt.show()
    # A[2]
    # plt.subplot(1, 2, 2)
    # file1 = "trained_models/Feb8th/user_loc0/best_weight/alpha=0p01/weighted_sumrate_best_weight_Nrf=8.npy"
    # file2 = "trained_models/Feb8th/user_loc0/best_weight/weighted_sumrate_best_weight_Nrf=8.npy"
    # a0p01 = np.load(file1).sum(axis=1).sum(axis=0)
    # a0p05 = np.load(file2).sum(axis=1).sum(axis=0)
    # plt.plot(a0p01)
    # plt.plot(a0p05)
    #
    # plt.show()
    # A[2]
    # Axes3D import has side effects, it enables using projection='3d' in add_subplot
    custome_obj = {'Closest_embedding_layer': Closest_embedding_layer, 'Interference_Input_modification': Interference_Input_modification,
                   'Interference_Input_modification_no_loop': Interference_Input_modification_no_loop,
                   "Interference_Input_modification_per_user":Interference_Input_modification_per_user,
                   "Closest_embedding_layer_moving_avg":Closest_embedding_layer_moving_avg,
                   "Per_link_Input_modification_more_G":Per_link_Input_modification_more_G,
                   "Per_link_Input_modification_more_G_less_X":Per_link_Input_modification_more_G_less_X,
                   "Per_link_Input_modification_even_more_G":Per_link_Input_modification_even_more_G,
                   "Per_link_Input_modification_compress_XG":Per_link_Input_modification_compress_XG,
                   "Per_link_Input_modification_compress_XG_alt": Per_link_Input_modification_compress_XG_alt,
                   "Per_link_Input_modification_more_G_alt_2":Per_link_Input_modification_more_G_alt_2,
                   "Per_link_Input_modification_compress_XG_alt_2":Per_link_Input_modification_compress_XG_alt_2,
                   "Per_link_Input_modification_most_G":Per_link_Input_modification_most_G,
                   "Per_link_sequential_modification": Per_link_sequential_modification,
                   "Per_link_sequential_modification_compressedX":Per_link_sequential_modification_compressedX,
                   "Per_link_Input_modification_most_G_raw_self":Per_link_Input_modification_most_G_raw_self,
                   "Reduced_output_input_mod":Reduced_output_input_mod,
                   "TopPrecoderPerUserInputMod":TopPrecoderPerUserInputMod,
                   "X_extends":X_extends,
                   "Per_link_Input_modification_most_G_col":Per_link_Input_modification_most_G_col,
                   "Sparsemax":Sparsemax,
                   "Sequential_Per_link_Input_modification_most_G_raw_self":Sequential_Per_link_Input_modification_most_G_raw_self,
                   "Per_link_Input_modification_most_G_raw_self_sigmoid":Per_link_Input_modification_most_G_raw_self_sigmoid,
                   "Per_link_Input_modification_most_G_raw_self_more_interference":Per_link_Input_modification_most_G_raw_self_more_interference,
                   "Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum":Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum,
                   "Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights":Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights,
                   "Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights_different_weights":Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights_different_weights}

    # greedy_grid_search()
    # training_data = np.load("trained_models\Dec_13\GNN_grid_search_temp=0.1.npy")
    # plot_dat

    file = "trained_models/Jan_18/30AOA/test_dnn_Nrf=8+annealing_LR.npy"
    # file = "trained_models/Nov_23/B=32_one_CE_loss/N_rf=1+VAEB=1x32E=4+1x512_per_linkx6_alt+CE_loss+MP"
    # for item in [0.01, 0.1, 1, 5, 10]:
    #     garsons_method(file.format(item))
    # obtain_channel_distributions(10000, 50, 64, 5)
    # A[2]
    N = 1
    M = 64
    K = 100
    B = 32
    seed = 200
    check = 100
    N_rf = 8
    sigma2_h = 6.3
    sigma2_n = 1
    tf.random.set_seed(seed)
    np.random.seed(seed)
    # greedy_grid_search()
    # A[2]
    # partial_feedback_and_DNN_grid_search()
    # compare_quantizers(1)
    model_path = "trained_models/Feb8th/user_loc0/train_with_0_1_weight/alpha=0p01_NRF={}.h5"
    mores = [8,7,6,5,4,3,2,1]
    Es = [1]
    #
    # model = DP_DNN_feedback_pure_greedy_model(N_rf, 32, 2, M, K, sigma2_n, perfect_CSI=True)
    # test_greedy(model, M, K, N_rf=8)
    # A[2]
    # test_greedy_weighted_SR(0, M=M, K=K, B=B, N_rf=N_rf, sigma2_n=sigma2_n, sigma2_h=sigma2_h)
    # A[2]
    # model = DP_partial_feedback_pure_greedy_model(32, 2, 2, M, K, sigma2_n, perfect_CSI=True)
    # test_greedy(model, M=M, K=K, B=B, N_rf=N_rf, sigma2_n=sigma2_n, sigma2_h=sigma2_h)
    for i in Es:
        for j in mores:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            N_rf = j
            links = j
            bits = i
            print("========================================== links =", j, "bits = ", i)
            # test_random_weighted_SR(0, M=M, K=K, B=B, N_rf=N_rf, sigma2_n=sigma2_n, sigma2_h=sigma2_h)
            test_greedy_weighted_SR(0, M=M, K=K, B=B, N_rf=N_rf, sigma2_n=sigma2_n, sigma2_h=sigma2_h)
            # test_PF_DFT_weighted_SR(0, M=M, K=K, B=B, N_rf=N_rf, sigma2_n=sigma2_n, sigma2_h=sigma2_h, alpha = 0.05)
            # test_BestWeight_weighted_SR(0, M=M, K=K, B=B, N_rf=N_rf, sigma2_n=sigma2_n, sigma2_h=sigma2_h)
            # dnn_model = tf.keras.models.load_model(model_path.format(N_rf), custom_objects=custome_obj)
            # test_performance_weighted_SR(dnn_model, M=M, K=K, B=B, N_rf=N_rf, sigma2_n=sigma2_n, sigma2_h=sigma2_h)
            # print(model.get_layer("model_2").get_layer("model_1").summary())
            # model = tf.keras.models.load_model(model_path, custom_objects=custome_obj)
            # model = partial_feedback_top_N_rf_model(N_rf, B, 1, M, K, sigma2_n)
            #     print(model.get_layer("model").summary())
            #     print(model.summary())
            # model = NN_Clustering(N_rf, M, reduced_dim=8)
            # model = top_N_rf_user_model(M, K, N_rf)
            # model = partial_feedback_pure_greedy_model_not_perfect_CSI_available(N_rf, 32, 10, M, K, sigma2_n)

            # model = relaxation_based_solver(M, K, N_rf)
            # garbage, g_max = Input_normalization_per_user(tf.abs(generate_link_channel_data(1000, K, M, Nrf=N_rf)))
            # feed_back_model = max_min_k_link_feedback_model(N_rf, bits, links, M, K)

            # test_random_weighted_SR(None, M, K, B, N_rf, 6.3, 1, printing=True)
            # model = DP_partial_feedback_pure_greedy_model_new_feedback_model(N_rf, 64, 10, M, K, sigma2_n, perfect_CSI=True)
            # test_performance_partial_feedback_and_DNN(feed_back_model, dnn_model, M=M, K=K, B=B, N_rf=N_rf, sigma2_n=sigma2_n, sigma2_h = sigma2_h)

            # test_greedy_weighted_SR(None, M, K, B, N_rf, 6.3, 1, printing=True)
            # dnn_model = tf.keras.models.load_model(model_path.format(N_rf), custom_objects=custome_obj)
            # test_performance_weighted_SR(dnn_model, M=M, K=K, B=B, N_rf=N_rf, sigma2_n=sigma2_n, sigma2_h = sigma2_h)
            # test_BestWeight_weighted_SR(None, M, K, B, N_rf)
            # test_DNN_different_K(model_path, M=M, K=K, B=B, N_rf=N_rf, sigma2_n=sigma2_n, sigma2_h = sigma2_h)
            # vvvvvvvvvvvvvvvvvv using dynamic programming to do N_rf sweep of Greedy faster vvvvvvvvvvvvvvvvvv
            # ^^^^^^^^^^^^^^^^^^ using dynamic programming to do N_rf sweep of Greedy faster ^^^^^^^^^^^^^^^^^^
            # test_greedy(M=M, K=K, B=B, N_rf=N_rf, sigma2_n=sigma2_n, sigma2_h = sigma2_h)
