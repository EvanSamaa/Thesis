import tensorflow as tf
from tf_agents.distributions import gumbel_softmax
from util import *
from models import *
import numpy as np
import scipy as sp
from relaxflow.reparam import CategoricalReparam
from relaxflow.relax import RELAX
from keras_adabound.optimizers import AdaBound
def grid_search_gumbel_sm(N_rf = 8):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template_template = "trained_models/Jan_13/GNN_annealing_temp_Nrf={}+180_half_AOE".format(N_rf)
    fname_template = fname_template_template + "{}"
    check = 250
    SUPERVISE_TIME = 0
    training_mode = 2
    swap_delay = check / 2

    # problem Definition
    M = 64
    K = 50
    seed = 100
    # N_rf = 8
    sigma2_h = 6.3
    sigma2_n = 1.0

    ############################### generate data ###############################
    valid_data = generate_link_channel_data_fullAOE(1000, K, M, Nrf=N_rf)
    from matplotlib import pyplot as plt
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    ################################ hyperparameters ###############################
    EPOCHS = 100000
    lr = 0.001
    N = 50 # number of
    rounds = 5
    sample_size = 100
    temp = 0.1
    check = 50
    model = FDD_per_link_archetecture_more_G(M, K, 5, N_rf, True, max_val)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    ################################ Metrics  ###############################
    sum_rate = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    train_sum_rate = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_hard_loss = tf.keras.metrics.Mean(name='train_loss')
    ################################ storing train data in npy file  ##############################
    # the three would be first train_loss, Hardloss, and the validation loss, every 50 iterations
    max_acc = 0
    np_data = ModelTrainer(save_dir=fname_template.format(".npy"), data_cols=3, epoch=EPOCHS)
    # training loop
    for i in range(0, EPOCHS):
        train_hard_loss.reset_states()
        # generate training data
        train_data = generate_link_channel_data_fullAOE(N, K, M, Nrf=N_rf)
        ###################### training happens here ######################
        for e in range(0, rounds):
            temp = 0.5 * np.exp(-4.5 / rounds * e) + 0.1
            temp = np.float32(temp)
            train_hard_loss.reset_states()
            train_loss.reset_states()
            with tf.GradientTape(persistent=True) as tape:
                ###################### model post-processing ######################
                ans, raw_ans = model(train_data) # raw_ans is in the shape of (N, passes, M*K, N_rf)
                raw_ans = tf.transpose(raw_ans, [0, 1, 3, 2])
                out_raw = tf.reshape(raw_ans[:,-1], [N * N_rf, K*M])
                sm = gumbel_softmax.GumbelSoftmax(temperature=temp, logits=out_raw)
                out_raw = sm.sample(sample_size)
                out_raw = tf.reshape(out_raw, [sample_size, N, N_rf, K*M])
                out = tf.reduce_sum(out_raw, axis=2)
                out = tf.reshape(out, [sample_size*N, K*M])
                train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [100,1, 1, 1]), [100*N, K, M])
                ###################### model post-processing ######################
                loss = train_sum_rate(out, train_label)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,model.trainable_variables))
            # optimizer.minimize(loss, ans)
            train_loss(loss)
            train_hard_loss(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(ans[:,-1]), train_data))
            print(train_hard_loss.result(),train_loss.result())
            del tape
        ###################### testing with validation set ######################
        if i%check == 0:
            scheduled_output, raw_output = model.predict(valid_data, batch_size=N)
            valid_loss = tf.reduce_mean(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(scheduled_output[:, -1]), valid_data))
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), valid_loss])
            print("============================================================\n")
            print(valid_loss)
            if valid_loss < max_acc:
                max_acc = valid_loss
                model.save(fname_template.format(".h5"))
            if i >= (check * 2):
                graphing_data = np_data.data
                improvement = graphing_data[i + 1 - (check * 2): i - check + 1, -1].min() - graphing_data[
                                                                                                   i - check + 1: i + 1,
                                                                                                   -1].min()
                counter = 0
                for asldk in graphing_data[0:i + 1, -1]:
                    if asldk != 0:
                        print(counter, asldk)
                    counter = counter + 1
                print("the improvement in the past 500 epochs is: ", improvement)
                print("the validation SR is: ", valid_loss)
                if improvement <= 0.0001 and lr == 0.001:
                    optimizer = tf.keras.optimizers.Adam(lr=0.0001)
                elif improvement <= 0.0001 and lr < 0.001:
                    break
        else:
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), 0])
    np_data.save()
def grid_search_relax(N_rf = 8):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template_template = "trained_models/Jan_18/test_dnn_Nrf={}0.05Base_temp".format(N_rf)
    fname_template = fname_template_template + "{}"
    check = 250
    SUPERVISE_TIME = 0
    training_mode = 2
    swap_delay = check / 2

    # problem Definition
    M = 64
    K = 50
    seed = 100
    # N_rf = 8
    sigma2_h = 6.3
    sigma2_n = 1.0

    ############################### generate data ###############################
    valid_data = generate_link_channel_data_fullAOE(1000, K, M, Nrf=N_rf)
    from matplotlib import pyplot as plt
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    ################################ hyperparameters ###############################
    EPOCHS = 100000
    lr = 0.001
    N = 50 # number of
    rounds = 15
    sample_size = 50
    temp = 0.1
    check = 50
    model = FDD_agent_more_G(M, K, 5, N_rf, True, max_val)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    ################################ Metrics  ###############################
    sum_rate = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    train_sum_rate = Sum_rate_utility_WeiCui_stable(K, M, sigma2_n)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_hard_loss = tf.keras.metrics.Mean(name='train_loss')
    ################################ storing train data in npy file  ##############################
    # the three would be first train_loss, Hardloss, and the validation loss, every 50 iterations
    max_acc = 0
    np_data = ModelTrainer(save_dir=fname_template.format(".npy"), data_cols=3, epoch=EPOCHS)
    # training loop
    for i in range(0, EPOCHS):
        train_hard_loss.reset_states()
        # generate training data
        train_data = generate_link_channel_data_fullAOE(N, K, M, Nrf=N_rf)
        ###################### training happens here ######################
        for e in range(0, rounds):
            temp = 0.5 * np.exp(-4.5 / rounds * e) * tf.maximum(0.0, ((200.0-i)/200.0)) + 0.1
            temp = np.float32(temp)
            train_hard_loss.reset_states()
            train_loss.reset_states()
            with tf.GradientTape(persistent=True) as tape:
                ###################### model post-processing ######################
                ans, raw_ans = model(train_data) # raw_ans is in the shape of (N, passes, M*K, N_rf)
                raw_ans = tf.transpose(raw_ans, [0, 1, 3, 2])
                out_raw = tf.reshape(raw_ans[:,-1], [N * N_rf, K*M])
                sm = gumbel_softmax.GumbelSoftmax(temperature=temp, logits=out_raw)
                out_raw = sm.sample(sample_size)
                out_raw = tf.reshape(out_raw, [sample_size, N, N_rf, K*M])
                out = tf.reduce_sum(out_raw, axis=2)
                out = tf.reshape(out, [sample_size*N, K*M])
                train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [sample_size,1, 1, 1]), [sample_size*N, K, M])
                ###################### model post-processing ######################
                loss = train_sum_rate(out, train_label)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,model.trainable_variables))
            # optimizer.minimize(loss, ans)
            train_loss(loss)
            train_hard_loss(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(ans[:,-1]), train_data))
            print(train_hard_loss.result(),train_loss.result(), "stable sr =",tf.reduce_mean(loss))
            del tape
        ###################### testing with validation set ######################
        if i%check == 0:
            scheduled_output, raw_output = model.predict(valid_data, batch_size=N)
            valid_loss = tf.reduce_mean(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(scheduled_output[:, -1]), valid_data))
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), valid_loss])
            print("============================================================\n")
            print(valid_loss)
            if valid_loss < max_acc:
                max_acc = valid_loss
                model.save(fname_template.format(".h5"))
            if i >= (check * 2):
                graphing_data = np_data.data
                improvement = graphing_data[i + 1 - (check * 2): i - check + 1, -1].min() - graphing_data[
                                                                                                   i - check + 1: i + 1,
                                                                                                   -1].min()
                counter = 0
                for asldk in graphing_data[0:i + 1, -1]:
                    if asldk != 0:
                        print(counter, asldk)
                    counter = counter + 1
                print("the improvement in the past 500 epochs is: ", improvement)
                print("the validation SR is: ", valid_loss)
                if improvement <= 0.0001:
                    break
        else:
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), 0])
    np_data.save()
def grid_search_with_mutex_loss(N_rf = 8):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template_template = "trained_models/Jan_18/30AOA/test_dnn_Nrf={}+annealing_LR+stable_loss".format(N_rf)
    fname_template = fname_template_template + "{}"
    check = 250
    SUPERVISE_TIME = 0
    training_mode = 2
    swap_delay = check / 2

    # problem Definition
    M = 64
    K = 50
    seed = 100
    # N_rf = 8
    sigma2_h = 6.3
    sigma2_n = 1.0

    ############################### generate data ###############################
    valid_data = generate_link_channel_data(1000, K, M, Nrf=N_rf)
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    ################################ hyperparameters ###############################
    EPOCHS = 100000
    lr = 0.001
    N = 50 # number of
    rounds = 15
    sample_size = 50
    temp = 0.1
    check = 100
    model = FDD_agent_more_G(M, K, 5, N_rf, True, max_val)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    ################################ Metrics  ###############################
    sum_rate = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    train_sum_rate = Sum_rate_utility_WeiCui_stable(K, M, sigma2_n)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_hard_loss = tf.keras.metrics.Mean(name='train_loss')
    mutex_loss_fn = mutex_loss(N_rf, M, K, N)
    ################################ storing train data in npy file  ##############################
    # the three would be first train_loss, Hardloss, and the validation loss, every 50 iterations
    max_acc = 0
    np_data = ModelTrainer(save_dir=fname_template.format(".npy"), data_cols=3, epoch=EPOCHS)
    # training loop
    for i in range(0, EPOCHS):
        train_hard_loss.reset_states()
        # generate training data
        train_data = generate_link_channel_data(N, K, M, Nrf=N_rf)
        ###################### training happens here ######################
        for e in range(0, rounds):
            temp = 0.5 * np.exp(-4.5 / rounds * e) * tf.maximum(0.0, ((200.0-i)/200.0)) + 0.1
            temp = np.float32(temp)
            train_hard_loss.reset_states()
            train_loss.reset_states()
            with tf.GradientTape(persistent=True) as tape:
                ###################### model post-processing ######################
                ans, raw_ans = model(train_data) # raw_ans is in the shape of (N, passes, M*K, N_rf)
                out_raw = tf.transpose(raw_ans, [0, 1, 3, 2])
                out_raw = tf.reshape(out_raw[:,-1], [N * N_rf, K*M])
                sm = gumbel_softmax.GumbelSoftmax(temperature=temp, logits=out_raw)
                out_raw = sm.sample(sample_size)
                out_raw = tf.reshape(out_raw, [sample_size, N, N_rf, K*M])
                out = tf.reduce_sum(out_raw, axis=2)
                out = tf.reshape(out, [sample_size*N, K*M])
                train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [sample_size,1, 1, 1]), [sample_size*N, K, M])
                ###################### model post-processing ######################
                loss = train_sum_rate(out, train_label) + mutex_loss_fn(raw_ans[:, -1])
                # loss = train_sum_rate(out, train_label) + 0.01 *mutex_loss_fn(out)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,model.trainable_variables))
            # optimizer.minimize(loss, ans)
            train_loss(loss)
            train_hard_loss(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(ans[:,-1]), train_data))
            print(train_hard_loss.result(),train_loss.result(), "stable sr = ", tf.reduce_mean(loss))
            del tape
        ###################### testing with validation set ######################
        if i%check == 0:
            scheduled_output, raw_output = model.predict(valid_data, batch_size=N)
            valid_loss = tf.reduce_mean(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(scheduled_output[:, -1]), valid_data))
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), valid_loss])
            print("============================================================\n")
            print(valid_loss)
            if valid_loss < max_acc:
                max_acc = valid_loss
                model.save(fname_template.format(".h5"))
            if i >= (check * 2):
                graphing_data = np_data.data
                improvement = graphing_data[i + 1 - (check * 2): i - check + 1, -1].min() - graphing_data[
                                                                                                   i - check + 1: i + 1,
                                                                                                   -1].min()
                counter = 0
                for asldk in graphing_data[0:i + 1, -1]:
                    if asldk != 0:
                        print(counter, asldk)
                    counter = counter + 1
                print("the improvement in the past 500 epochs is: ", improvement)
                print("the validation SR is: ", valid_loss)
                if improvement <= 0.0001 and lr == 0.001:
                    lr = 0.0001
                    optimizer = tf.keras.optimizers.Adam(lr=0.0001)
                elif improvement <= 0.0001 and lr < 0.001:
                    break
        else:
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), 0])
    np_data.save()
def grid_search_with_Nto1(N_rf = 8):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template_template = "trained_models/Jan_18/test_DNN_Nto1_Nrf={}".format(N_rf)
    fname_template = fname_template_template + "{}"
    check = 250
    SUPERVISE_TIME = 0
    training_mode = 2
    swap_delay = check / 2

    # problem Definition
    M = 64
    K = 50
    seed = 100
    # N_rf = 8
    sigma2_h = 6.3
    sigma2_n = 1.0

    ############################### generate data ###############################
    valid_data = generate_link_channel_data_fullAOE(1000, K, M, Nrf=N_rf)
    from matplotlib import pyplot as plt
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    ################################ hyperparameters ###############################
    EPOCHS = 100000
    lr = 0.001
    N = 50 # number of
    rounds = 15
    sample_size = 50
    temp = 0.1
    check = 100
    model = FDD_agent_more_G_with_moderator(M, K, 5, N_rf, True, max_val)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    ################################ Metrics  ###############################
    sum_rate = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    train_sum_rate = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_hard_loss = tf.keras.metrics.Mean(name='train_loss')
    mutex_loss_fn = mutex_loss(N_rf, M, K, N)
    ################################ storing train data in npy file  ##############################
    # the three would be first train_loss, Hardloss, and the validation loss, every 50 iterations
    max_acc = 0
    np_data = ModelTrainer(save_dir=fname_template.format(".npy"), data_cols=3, epoch=EPOCHS)
    # training loop
    for i in range(0, EPOCHS):
        train_hard_loss.reset_states()
        # generate training data
        train_data = generate_link_channel_data_fullAOE(N, K, M, Nrf=N_rf)
        ###################### training happens here ######################
        for e in range(0, rounds):
            temp = 0.5 * np.exp(-4.5 / rounds * e) * tf.maximum(0.0, ((200.0-i)/200.0)) + 0.1
            temp = np.float32(temp)
            train_hard_loss.reset_states()
            train_loss.reset_states()
            with tf.GradientTape(persistent=True) as tape:
                ###################### model post-processing ######################
                ans, raw_ans = model(train_data) # raw_ans is in the shape of (N, passes, M*K, N_rf)
                out_raw = tf.transpose(raw_ans, [0, 1, 3, 2])
                out_raw = tf.reshape(out_raw[:,-1], [N * N_rf, K*M])
                sm = gumbel_softmax.GumbelSoftmax(temperature=temp, logits=out_raw)
                out_raw = sm.sample(sample_size)
                out_raw = tf.reshape(out_raw, [sample_size, N, N_rf, K*M])
                out = tf.reduce_sum(out_raw, axis=2)
                out = tf.reshape(out, [sample_size*N, K*M])
                train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [sample_size,1, 1, 1]), [sample_size*N, K, M])
                ###################### model post-processing ######################
                loss = train_sum_rate(out, train_label) + mutex_loss_fn(raw_ans[:, -1])
                # loss = train_sum_rate(out, train_label) + 0.01 *mutex_loss_fn(out)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,model.trainable_variables))
            # optimizer.minimize(loss, ans)
            train_loss(loss)
            train_hard_loss(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(ans[:,-1]), train_data))
            print(train_hard_loss.result(),train_loss.result())
            del tape
        ###################### testing with validation set ######################
        if i%check == 0:
            scheduled_output, raw_output = model.predict(valid_data, batch_size=N)
            valid_loss = tf.reduce_mean(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(scheduled_output[:, -1]), valid_data))
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), valid_loss])
            print("============================================================\n")
            print(valid_loss)
            if valid_loss < max_acc:
                max_acc = valid_loss
                model.save(fname_template.format(".h5"))
            if i >= (check * 2):
                graphing_data = np_data.data
                improvement = graphing_data[i + 1 - (check * 2): i - check + 1, -1].min() - graphing_data[
                                                                                                   i - check + 1: i + 1,
                                                                                                   -1].min()
                counter = 0
                for asldk in graphing_data[0:i + 1, -1]:
                    if asldk != 0:
                        print(counter, asldk)
                    counter = counter + 1
                print("the improvement in the past 500 epochs is: ", improvement)
                print("the validation SR is: ", valid_loss)
                if improvement <= 0.0001:
                    break
        else:
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), 0])
    np_data.save()
def grid_search_REBAR(N_rf = 8):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template_template = "trained_models/Jan_18/RELAX_Nrf={}".format(N_rf)
    fname_template = fname_template_template + "{}"
    check = 250
    SUPERVISE_TIME = 0
    training_mode = 2
    swap_delay = check / 2

    # problem Definition
    M = 64
    K = 50
    seed = 100
    # N_rf = 8
    sigma2_h = 6.3
    sigma2_n = 1.0


    ############################### generate data ###############################
    valid_data = generate_link_channel_data_fullAOE(1000, K, M, Nrf=N_rf)
    from matplotlib import pyplot as plt
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    ################################ hyperparameters ###############################
    EPOCHS = 100000
    lr = 0.001
    N = 50 # number of
    rounds = 5
    sample_size = 10
    temp = 0.1
    check = 100
    model = FDD_agent_more_G(M, K, 5, N_rf, True, max_val)
    # model = tf.Variable(tf.ones((N, M*K, 8)), dtype = tf.float32)
    nu = tf.Variable(0.1, dtype = tf.float32)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    ################################ Metrics  ###############################
    sum_rate = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    train_sum_rate = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_hard_loss = tf.keras.metrics.Mean(name='train_loss')
    mutex_loss_fn = mutex_loss(N_rf, M, K, N)
    def custom_loss_fun(train_data, nu):
        @tf.custom_gradient
        def out_put_func(raw_ans):
            def grad(dy):
                out_raw = tf.transpose(raw_ans[:, -1], [0, 2, 1])
                out_raw = tf.tile(out_raw, [sample_size, 1, 1])
                out_raw = tf.reshape(out_raw, [N * sample_size * N_rf, K * M])
                rep = CategoricalReparam(out_raw)
                train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [sample_size, 1, 1, 1]),
                                         [sample_size * N, K, M])
                tep = RELAX(tape, train_sum_rate, N * sample_size, N_rf, K, M, train_label,
                            *rep.rebar_params(train_sum_rate, nu, train_label, N * sample_size, N_rf, K, M), [out_raw],
                            params=model.trainable_variables, var_params=[nu])
                print(tep)
                return dy * tep

            return 1.0, grad
        return out_put_func
    ################################ storing train data in npy file  ##############################
    # the three would be first train_loss, Hardloss, and the validation loss, every 50 iterations
    max_acc = 0
    np_data = ModelTrainer(save_dir=fname_template.format(".npy"), data_cols=3, epoch=EPOCHS)
    # training loop
    for i in range(0, EPOCHS):
        train_hard_loss.reset_states()
        # generate training data
        train_data = generate_link_channel_data_fullAOE(N, K, M, Nrf=N_rf)
        ###################### training happens here ######################
        for e in range(0, rounds):
            temp = 0.5 * np.exp(-4.5 / rounds * e) * tf.maximum(0.0, ((200.0-i)/200.0)) + 0.1
            temp = np.float32(temp)
            train_hard_loss.reset_states()
            train_loss.reset_states()
            with tf.GradientTape(persistent=True) as tape:
                ###################### model post-processing ######################
                # print(model.shape)
                ans, raw_ans = model(train_data) # raw_ans is in the shape of (N, passes, M*K, N_rf)
                loss = custom_loss_fun(train_data, nu)(raw_ans)

                # out_raw = tf.transpose(raw_ans[:,-1], [0, 2, 1])
                # out_raw = tf.tile(out_raw, [sample_size, 1, 1])
                # out_raw = tf.reshape(out_raw, [N * sample_size * N_rf, K * M])
                # rep = CategoricalReparam(out_raw)
                # train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [sample_size, 1, 1, 1]), [sample_size * N, K, M])
                # grad = RELAX(tape, train_sum_rate, N*sample_size, N_rf, K, M, train_label,
                #              *rep.rebar_params(train_sum_rate, nu, train_label, N*sample_size, N_rf, K, M), [out_raw], params=model.trainable_variables, var_params=[nu])

                ###################### model post-processing ######################
                # loss = train_sum_rate(ans[:,-1], train_data) + 0.01*mutex_loss_fn(raw_ans[:, -1])
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            # optimizer.minimize(loss, ans)
            train_loss(loss)
            train_hard_loss(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(ans[:,-1]), train_data))
            print(train_hard_loss.result(),train_loss.result())
            del tape
        ###################### testing with validation set ######################
        if i%check == 0:
            scheduled_output, raw_output = model.predict(valid_data, batch_size=N)
            valid_loss = tf.reduce_mean(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(scheduled_output[:, -1]), valid_data))
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), valid_loss])
            print("============================================================\n")
            print(valid_loss)
            if valid_loss < max_acc:
                max_acc = valid_loss
                model.save(fname_template.format(".h5"))
            if i >= (check * 2):
                graphing_data = np_data.data
                improvement = graphing_data[i + 1 - (check * 2): i - check + 1, -1].min() - graphing_data[
                                                                                                   i - check + 1: i + 1,
                                                                                                   -1].min()
                counter = 0
                for asldk in graphing_data[0:i + 1, -1]:
                    if asldk != 0:
                        print(counter, asldk)
                    counter = counter + 1
                print("the improvement in the past 500 epochs is: ", improvement)
                print("the validation SR is: ", valid_loss)
                if improvement <= 0.0001:
                    break
        else:
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), 0])
    np_data.save()
def grid_search_with_emsemble(N_rf = 8):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template_template = "trained_models/Jan_18/test_DNN_ensemble_Nrf={}".format(N_rf)
    fname_template = fname_template_template + "{}"
    check = 250
    SUPERVISE_TIME = 0
    training_mode = 2
    swap_delay = check / 2

    # problem Definition
    M = 64
    K = 50
    seed = 100
    # N_rf = 8
    sigma2_h = 6.3
    sigma2_n = 1.0

    ############################### generate data ###############################
    valid_data = generate_link_channel_data_fullAOE(1000, K, M, Nrf=N_rf)
    from matplotlib import pyplot as plt
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    ################################ hyperparameters ###############################
    EPOCHS = 100000
    lr = 0.001
    N = 25 # number of
    rounds = 5
    sample_size = 50
    temp = 0.1
    check = 100
    model = FDD_ensemble_model(M, K, 5, N_rf, True, max_val)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    ################################ Metrics  ###############################
    sum_rate = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    train_sum_rate = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_hard_loss = tf.keras.metrics.Mean(name='train_loss')
    mutex_loss_fn = mutex_loss(N_rf, M, K, N)
    ################################ storing train data in npy file  ##############################
    # the three would be first train_loss, Hardloss, and the validation loss, every 50 iterations
    max_acc = 0
    np_data = ModelTrainer(save_dir=fname_template.format(".npy"), data_cols=3, epoch=EPOCHS)
    # training loop
    for i in range(0, EPOCHS):
        train_hard_loss.reset_states()
        # generate training data
        train_data = generate_link_channel_data_fullAOE(N, K, M, Nrf=N_rf)
        ###################### training happens here ######################
        for e in range(0, rounds):
            temp = 0.5 * np.exp(-4.5 / rounds * e) * tf.maximum(0.0, ((200.0-i)/200.0)) + 0.1
            temp = np.float32(temp)
            train_hard_loss.reset_states()
            train_loss.reset_states()
            with tf.GradientTape(persistent=True) as tape:
                ###################### model post-processing ######################
                ans1, raw_ans1, ans2, raw_ans2 = model(train_data) # raw_ans is in the shape of (N, passes, M*K, N_rf)
                out_raw1 = tf.transpose(raw_ans1, [0, 1, 3, 2])
                out_raw1 = tf.reshape(out_raw1[:,-1], [N * N_rf, K*M])
                sm1 = gumbel_softmax.GumbelSoftmax(temperature=temp, logits=out_raw1)
                out_raw1 = sm1.sample(sample_size)
                out_raw1 = tf.reshape(out_raw1, [sample_size, N, N_rf, K*M])
                out1 = tf.reduce_sum(out_raw1, axis=2)
                out1 = tf.reshape(out1, [sample_size*N, K*M])
                train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [sample_size,1, 1, 1]), [sample_size*N, K, M])
                ###################### model post-processing ######################
                loss = train_sum_rate(out1, train_label) + 0.01*mutex_loss_fn(raw_ans1[:, -1])

                out_raw2 = tf.transpose(raw_ans2, [0, 1, 3, 2])
                out_raw2 = tf.reshape(out_raw2[:, -1], [N * N_rf, K * M])
                sm2 = gumbel_softmax.GumbelSoftmax(temperature=temp, logits=out_raw2)
                out_raw2 = sm2.sample(sample_size)
                out_raw2 = tf.reshape(out_raw2, [sample_size, N, N_rf, K * M])
                out2 = tf.reduce_sum(out_raw2, axis=2)
                out2 = tf.reshape(out2, [sample_size * N, K * M])

                loss = loss + train_sum_rate(out2, train_label) + 0.01 * mutex_loss_fn(raw_ans2[:, -1])
                # loss = train_sum_rate(out, train_label) + 0.01 *mutex_loss_fn(out)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,model.trainable_variables))
            # optimizer.minimize(loss, ans)
            train_loss(loss/2)
            train_hard_loss(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(ans2[:,-1]), train_data))
            print(train_hard_loss.result(),train_loss.result())
            del tape
        ###################### testing with validation set ######################
        if i%check == 0:
            scheduled_output, raw_ans1, ans2, raw_ans2 = model.predict(valid_data, batch_size=N)
            valid_loss = tf.reduce_mean(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(scheduled_output[:, -1]), valid_data))
            valid_loss = tf.minimum(valid_loss, tf.reduce_mean(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(ans2[:, -1]), valid_data)))
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), valid_loss])
            print("============================================================\n")
            print(valid_loss)
            if valid_loss < max_acc:
                max_acc = valid_loss
                model.save(fname_template.format(".h5"))
            if i >= (check * 2):
                graphing_data = np_data.data
                improvement = graphing_data[i + 1 - (check * 2): i - check + 1, -1].min() - graphing_data[
                                                                                                   i - check + 1: i + 1,
                                                                                                   -1].min()
                counter = 0
                for asldk in graphing_data[0:i + 1, -1]:
                    if asldk != 0:
                        print(counter, asldk)
                    counter = counter + 1
                print("the improvement in the past 500 epochs is: ", improvement)
                print("the validation SR is: ", valid_loss)
                if improvement <= 0.0001:
                    break
        else:
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), 0])
    np_data.save()

if __name__ == "__main__":
    for N_rf_to_search in [8,7,6,5,4,3,2,1]:
        grid_search_with_mutex_loss(N_rf_to_search)
