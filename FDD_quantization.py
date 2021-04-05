import tensorflow as tf
from tf_agents.distributions import gumbel_softmax
from util import *
from models import *
import numpy as np
import scipy as sp
from keras_adabound.optimizers import AdaBound
def grid_search_student_teacher(bits = 8):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template_template = "trained_models/better_quantizer/student_teacher+tanh_{}bits"
    fname_template = fname_template_template.format(bits) + "{}"
    check = 250
    SUPERVISE_TIME = 0
    training_mode = 2
    swap_delay = check / 2

    # problem Definition
    M = 64
    K = 50
    B = 4
    E = 30
    more = bits
    seed = 100
    N_rf = 8
    sigma2_h = 6.3
    sigma2_n = 1.0
    res = 6
    ############################### generate data ###############################
    valid_data = generate_link_channel_data(1000, K, M, Nrf=N_rf)
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    q_valid_data = tf.abs(valid_data) / max_val
    q_valid_data = tf.where(q_valid_data > 1.0, 1.0, q_valid_data)
    q_valid_data = tf.round(q_valid_data * (2 ** res - 1)) / (2 ** res - 1) * max_val
    ################################ hyperparameters ###############################
    EPOCHS = 1000000
    lr = 0.001
    N = 400 # number of
    rounds = 1000
    sample_size = 100
    temp = 0.1
    check = 200
    model = CSI_reconstruction_knowledge_distillation(M, K, B, E, N_rf, more=more, avg_max=max_val)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    ################################ Metrics  ###############################
    sum_rate = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    train_reconstruction_loss_student = tf.keras.metrics.Mean(name='train_loss')
    train_reconstruction_loss_teacher = tf.keras.metrics.Mean(name='train_loss')
    ################################ storing train data in npy file  ##############################
    # the three would be first train_loss, Hardloss, and the validation loss, every 50 iterations
    max_acc = 10000
    np_data = ModelTrainer(save_dir=fname_template.format(".npy"), data_cols=3, epoch=EPOCHS)
    # training loop
    for i in range(0, EPOCHS):
        # generate training data
        train_reconstruction_loss_teacher.reset_states()
        train_reconstruction_loss_student.reset_states()
        train_data = generate_link_channel_data(N, K, M, Nrf=N_rf)
        ###################### training happens here ######################
        for e in range(0, 1):
            train_reconstruction_loss_student.reset_states()
            with tf.GradientTape(persistent=True) as tape:
                ###################### model post-processing ######################
                # q_train_data = tf.abs(train_data)/max_val
                # q_train_data = tf.where(q_train_data > 1.0, 1.0, q_train_data)
                # q_train_data = tf.round(q_train_data * (2 ** res - 1)) / (2 ** res - 1) * max_val
                reconstructed_input_teacher, reconstructed_input = model(train_data) # raw_ans is in the shape of (N, passes, M*K, N_rf)
                train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [100,1, 1, 1]), [100*N, K, M])
                ###################### model post-processing ######################
                loss_teacher = tf.keras.losses.MeanSquaredError()(reconstructed_input_teacher, tf.abs(train_data))
                loss_student = tf.keras.losses.MeanSquaredError()(reconstructed_input, tf.abs(train_data))
            model_student = model.get_layer('encoder_0').trainable_variables + model.get_layer('decoder_0').trainable_variables
            model_teacher = model.get_layer('encoder_0').trainable_variables + model.get_layer('decoder_1').trainable_variables
            gradients_student = tape.gradient(loss_student, model_student)
            gradients_teacher = tape.gradient(loss_teacher, model_teacher)
            optimizer.apply_gradients(zip(gradients_teacher, model_teacher))
            optimizer.apply_gradients(zip(gradients_student, model_student))
            # optimizer.minimize(loss, ans)
            train_reconstruction_loss_student(loss_student)
            train_reconstruction_loss_teacher(loss_teacher)

            print(train_reconstruction_loss_teacher.result(), train_reconstruction_loss_student.result())
            del tape
        ###################### testing with validation set ######################
        if i%check == 0:
            reconstructed_input_teacher, reconstructed_input = model(valid_data)
            valid_loss = tf.keras.losses.MeanSquaredError()(reconstructed_input, tf.abs(valid_data))
            np_data.log(i, [train_reconstruction_loss_teacher.result(), train_reconstruction_loss_student.result(), valid_loss])
            print("============================================================\n")
            print(valid_loss)
            if valid_loss < max_acc:
                max_acc = valid_loss
                model.save(fname_template.format(".h5"))
            if i >= (check * 2):
                graphing_data = np_data.data
                improvement = graphing_data[i + 1 - (check * 2): i - check + 1, -1].mean()*check - graphing_data[
                                                                                                   i - check + 1: i + 1,
                                                                                                   -1].mean()*check
                counter = 0
                for asldk in graphing_data[0:i + 1, -1]:
                    if asldk != 0:
                        print(counter, asldk)
                    counter = counter + 1
                print("the improvement in the past {} epochs is: ".format(check), improvement)
                print("the validation SR is: ", valid_loss)
                if improvement <= 0.0001:
                    break
        else:
            np_data.log(i, [train_reconstruction_loss_teacher.result(), train_reconstruction_loss_student.result(), 0])
    np_data.save()
def grid_search_STE(bits = 8):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template_template = "trained_models/better_quantizer/ste_models/STE_{}bits"
    fname_template = fname_template_template.format(bits) + "{}"
    check = 250
    SUPERVISE_TIME = 0
    training_mode = 2
    swap_delay = check / 2

    # problem Definition
    M = 64
    K = 50
    B = 4
    E = 30
    more = bits
    seed = 100
    N_rf = 8
    sigma2_h = 6.3
    sigma2_n = 1.0
    res = 6
    ############################### generate data ###############################
    valid_data = generate_link_channel_data(1000, K, M, Nrf=N_rf)
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    q_valid_data = tf.abs(valid_data) / max_val
    q_valid_data = tf.where(q_valid_data > 1.0, 1.0, q_valid_data)
    q_valid_data = tf.round(q_valid_data * (2 ** res - 1)) / (2 ** res - 1) * max_val
    ################################ hyperparameters ###############################
    EPOCHS = 1000000
    lr = 0.001
    N = 400 # number of
    rounds = 1000
    sample_size = 100
    temp = 0.1
    check = 200
    model = CSI_reconstruction_model_seperate_decoders_naive(M, K, B, E, N_rf, more=more, avg_max=max_val)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    ################################ Metrics  ###############################
    sum_rate = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    train_reconstruction_loss_student = tf.keras.metrics.Mean(name='train_loss')
    ################################ storing train data in npy file  ##############################
    # the three would be first train_loss, Hardloss, and the validation loss, every 50 iterations
    max_acc = 10000
    np_data = ModelTrainer(save_dir=fname_template.format(".npy"), data_cols=2, epoch=EPOCHS)
    # training loop
    for i in range(0, EPOCHS):
        # generate training data
        train_reconstruction_loss_student.reset_states()
        train_data = generate_link_channel_data(N, K, M, Nrf=N_rf)
        ###################### training happens here ######################
        for e in range(0, 1):
            train_reconstruction_loss_student.reset_states()
            with tf.GradientTape(persistent=True) as tape:
                ###################### model post-processing ######################
                # q_train_data = tf.abs(train_data)/max_val
                # q_train_data = tf.where(q_train_data > 1.0, 1.0, q_train_data)
                # q_train_data = tf.round(q_train_data * (2 ** res - 1)) / (2 ** res - 1) * max_val
                reconstructed_input = model(train_data) # raw_ans is in the shape of (N, passes, M*K, N_rf)
                train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [100,1, 1, 1]), [100*N, K, M])
                ###################### model post-processing ######################
                loss_student = tf.keras.losses.MeanSquaredError()(reconstructed_input, tf.abs(train_data))
            gradients_student = tape.gradient(loss_student, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients_student, model.trainable_variables))
            # optimizer.minimize(loss, ans)
            train_reconstruction_loss_student(loss_student)

            print(train_reconstruction_loss_student.result())
            del tape
        ###################### testing with validation set ######################
        if i%check == 0:
            reconstructed_input = model(valid_data)
            valid_loss = tf.keras.losses.MeanSquaredError()(reconstructed_input, tf.abs(valid_data))
            np_data.log(i, [train_reconstruction_loss_student.result(), valid_loss])
            print("============================================================\n")
            print(valid_loss)
            if valid_loss < max_acc:
                max_acc = valid_loss
                model.save(fname_template.format(".h5"))
            if i >= (check * 2):
                graphing_data = np_data.data
                improvement = graphing_data[i + 1 - (check * 2): i - check + 1, -1].mean()*check - graphing_data[
                                                                                                   i - check + 1: i + 1,
                                                                                                   -1].mean()*check
                counter = 0
                for asldk in graphing_data[0:i + 1, -1]:
                    if asldk != 0:
                        print(counter, asldk)
                    counter = counter + 1
                print("the improvement in the past {} epochs is: ".format(check), improvement)
                print("the validation SR is: ", valid_loss)
                if improvement <= 0.0001:
                    break
        else:
            np_data.log(i, [train_reconstruction_loss_student.result(), 0])
    np_data.save()
def grid_search_VQVAE(bits = 8):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template_template = "trained_models/better_quantizer/VQVAE_{}bits"
    fname_template = fname_template_template.format(bits) + "{}"
    check = 250
    SUPERVISE_TIME = 0
    training_mode = 2
    swap_delay = check / 2

    # problem Definition
    M = 64
    K = 50
    B = 4
    E = 30
    more = bits
    seed = 100
    N_rf = 8
    sigma2_h = 6.3
    sigma2_n = 1.0
    res = 6
    ############################### generate data ###############################
    valid_data = generate_link_channel_data(1000, K, M, Nrf=N_rf)
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    q_valid_data = tf.abs(valid_data) / max_val
    q_valid_data = tf.where(q_valid_data > 1.0, 1.0, q_valid_data)
    q_valid_data = tf.round(q_valid_data * (2 ** res - 1)) / (2 ** res - 1) * max_val
    ################################ hyperparameters ###############################
    EPOCHS = 1000000
    lr = 0.001
    N = 400 # number of
    rounds = 1000
    sample_size = 100
    temp = 0.1
    check = 200
    model = CSI_reconstruction_model_seperate_decoders(M, K, B, E, N_rf, 6, more=int(more/B), avg_max=max_val)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    ################################ Metrics  ###############################
    sum_rate = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    train_reconstruction_loss_student = tf.keras.metrics.Mean(name='train_loss')
    vqvae_loss = VAE_loss_general()
    ################################ storing train data in npy file  ##############################
    # the three would be first train_loss, Hardloss, and the validation loss, every 50 iterations
    max_acc = 10000
    np_data = ModelTrainer(save_dir=fname_template.format(".npy"), data_cols=2, epoch=EPOCHS)
    # training loop
    for i in range(0, EPOCHS):
        # generate training data
        train_reconstruction_loss_student.reset_states()
        train_data = generate_link_channel_data(N, K, M, Nrf=N_rf)
        ###################### training happens here ######################
        for e in range(0, 1):
            train_reconstruction_loss_student.reset_states()
            with tf.GradientTape(persistent=True) as tape:
                ###################### model post-processing ######################
                # q_train_data = tf.abs(train_data)/max_val
                # q_train_data = tf.where(q_train_data > 1.0, 1.0, q_train_data)
                # q_train_data = tf.round(q_train_data * (2 ** res - 1)) / (2 ** res - 1) * max_val
                reconstructed_input, z_qq, z_e = model(train_data) # raw_ans is in the shape of (N, passes, M*K, N_rf)
                train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [100,1, 1, 1]), [100*N, K, M])
                ###################### model post-processing ######################
                mse_loss = tf.keras.losses.MeanSquaredError()(reconstructed_input, tf.abs(train_data))
                loss_student = mse_loss + vqvae_loss.call(z_qq, z_e)

            gradients_student = tape.gradient(loss_student, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients_student, model.trainable_variables))
            # optimizer.minimize(loss, ans)
            train_reconstruction_loss_student(mse_loss)

            print(train_reconstruction_loss_student.result())
            del tape
        ###################### testing with validation set ######################
        if i%check == 0:
            reconstructed_input, z_qq, z_e = model(valid_data)
            valid_loss = tf.keras.losses.MeanSquaredError()(reconstructed_input, tf.abs(valid_data))
            np_data.log(i, [train_reconstruction_loss_student.result(), valid_loss])
            print("============================================================\n")
            print(valid_loss)
            if valid_loss < max_acc:
                max_acc = valid_loss
                model.save(fname_template.format(".h5"))
            if i >= (check * 2):
                graphing_data = np_data.data
                improvement = graphing_data[i + 1 - (check * 2): i - check + 1, -1].mean()*check - graphing_data[
                                                                                                   i - check + 1: i + 1,
                                                                                                   -1].mean()*check
                counter = 0
                for asldk in graphing_data[0:i + 1, -1]:
                    if asldk != 0:
                        print(counter, asldk)
                    counter = counter + 1
                print("the improvement in the past {} epochs is: ".format(check), improvement)
                print("the validation SR is: ", valid_loss)
                if improvement <= 0.0001:
                    break
        else:
            np_data.log(i, [train_reconstruction_loss_student.result(), 0])
    np_data.save()
def grid_search_CNN(bits = 8):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template_template = "trained_models/better_quantizer/chunky_{}bits_2splits"
    fname_template = fname_template_template.format(bits) + "{}"
    check = 250
    SUPERVISE_TIME = 0
    training_mode = 2
    swap_delay = check / 2

    # problem Definition
    M = 64
    K = 50
    B = 4
    E = 30
    more = bits
    seed = 100
    N_rf = 8
    sigma2_h = 6.3
    sigma2_n = 1.0
    res = 6
    ############################### generate data ###############################
    valid_data = generate_link_channel_data(1000, K, M, Nrf=N_rf)
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    q_valid_data = tf.abs(valid_data) / max_val
    q_valid_data = tf.where(q_valid_data > 1.0, 1.0, q_valid_data)
    q_valid_data = tf.round(q_valid_data * (2 ** res - 1)) / (2 ** res - 1) * max_val
    ################################ hyperparameters ###############################
    EPOCHS = 1000000
    lr = 0.001
    N = 400 # number of
    rounds = 1000
    sample_size = 100
    temp = 0.1
    check = 200
    model = CSI_reconstruction_model_seperate_decoders_chunky(M, K, B, E, N_rf, more=more, avg_max=max_val)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    ################################ Metrics  ###############################
    train_reconstruction_loss_student = tf.keras.metrics.Mean(name='train_loss')
    ################################ storing train data in npy file  ##############################
    # the three would be first train_loss, Hardloss, and the validation loss, every 50 iterations
    max_acc = 10000
    np_data = ModelTrainer(save_dir=fname_template.format(".npy"), data_cols=2, epoch=EPOCHS)
    # training loop
    for i in range(0, EPOCHS):
        # generate training data
        train_reconstruction_loss_student.reset_states()
        train_data = generate_link_channel_data(N, K, M, Nrf=N_rf)
        ###################### training happens here ######################
        for e in range(0, 1):
            train_reconstruction_loss_student.reset_states()
            with tf.GradientTape(persistent=True) as tape:
                ###################### model post-processing ######################
                # q_train_data = tf.abs(train_data)/max_val
                # q_train_data = tf.where(q_train_data > 1.0, 1.0, q_train_data)
                # q_train_data = tf.round(q_train_data * (2 ** res - 1)) / (2 ** res - 1) * max_val
                reconstructed_input = model(train_data) # raw_ans is in the shape of (N, passes, M*K, N_rf)
                ###################### model post-processing ######################
                loss_student = tf.keras.losses.MeanSquaredError()(reconstructed_input, tf.abs(train_data))
            gradients_student = tape.gradient(loss_student, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients_student, model.trainable_variables))
            # optimizer.minimize(loss, ans)
            train_reconstruction_loss_student(loss_student)

            print(train_reconstruction_loss_student.result())
            del tape
        ###################### testing with validation set ######################
        if i%check == 0:
            reconstructed_input = model(valid_data)
            valid_loss = tf.keras.losses.MeanSquaredError()(reconstructed_input, tf.abs(valid_data))
            np_data.log(i, [train_reconstruction_loss_student.result(), valid_loss])
            print("============================================================\n")
            print(valid_loss)
            if valid_loss < max_acc:
                max_acc = valid_loss
                model.save(fname_template.format(".h5"))
            if i >= (check * 2):
                graphing_data = np_data.data
                improvement = graphing_data[i + 1 - (check * 2): i - check + 1, -1].mean()*check - graphing_data[
                                                                                                   i - check + 1: i + 1,
                                                                                                   -1].mean()*check
                counter = 0
                for asldk in graphing_data[0:i + 1, -1]:
                    if asldk != 0:
                        print(counter, asldk)
                    counter = counter + 1
                print("the improvement in the past {} epochs is: ".format(check), improvement)
                print("the validation SR is: ", valid_loss)
                if improvement <= 0.0001:
                    break
        else:
            np_data.log(i, [train_reconstruction_loss_student.result(), 0])
    np_data.save()
    tf.keras.backend.clear_session()
def grid_search_layers(bits = 8):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template_template = "trained_models/better_quantizer/layers_{}bits_max32bits"
    fname_template = fname_template_template.format(bits) + "{}"
    check = 250
    SUPERVISE_TIME = 0
    training_mode = 2
    swap_delay = check / 2

    # problem Definition
    M = 64
    K = 50
    B = 4
    E = 30
    more = bits
    seed = 100
    N_rf = 8
    sigma2_h = 6.3
    sigma2_n = 1.0
    res = 6
    ############################### generate data ###############################
    valid_data = generate_link_channel_data(1000, K, M, Nrf=N_rf)
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    q_valid_data = tf.abs(valid_data) / max_val
    q_valid_data = tf.where(q_valid_data > 1.0, 1.0, q_valid_data)
    q_valid_data = tf.round(q_valid_data * (2 ** res - 1)) / (2 ** res - 1) * max_val
    ################################ hyperparameters ###############################
    EPOCHS = 1000000
    lr = 0.001
    N = 400 # number of
    rounds = 1000
    sample_size = 100
    temp = 0.1
    check = 200
    model = CSI_reconstruction_model_seperate_decoders_layers(M, K, B, E, N_rf, more=more, avg_max=max_val)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    ################################ Metrics  ###############################
    train_reconstruction_loss_student = tf.keras.metrics.Mean(name='train_loss')
    ################################ storing train data in npy file  ##############################
    # the three would be first train_loss, Hardloss, and the validation loss, every 50 iterations
    max_acc = 10000
    np_data = ModelTrainer(save_dir=fname_template.format(".npy"), data_cols=2, epoch=EPOCHS)
    # training loop
    for i in range(0, EPOCHS):
        # generate training data
        train_reconstruction_loss_student.reset_states()
        train_data = generate_link_channel_data(N, K, M, Nrf=N_rf)
        ###################### training happens here ######################
        for e in range(0, 1):
            train_reconstruction_loss_student.reset_states()
            with tf.GradientTape(persistent=True) as tape:
                ###################### model post-processing ######################
                # q_train_data = tf.abs(train_data)/max_val
                # q_train_data = tf.where(q_train_data > 1.0, 1.0, q_train_data)
                # q_train_data = tf.round(q_train_data * (2 ** res - 1)) / (2 ** res - 1) * max_val
                reconstructed_input = model(train_data) # raw_ans is in the shape of (N, passes, M*K, N_rf)
                ###################### model post-processing ######################
                loss_student = tf.keras.losses.MeanSquaredError()(reconstructed_input, tf.abs(train_data))
            gradients_student = tape.gradient(loss_student, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients_student, model.trainable_variables))
            # optimizer.minimize(loss, ans)
            train_reconstruction_loss_student(loss_student)

            print(train_reconstruction_loss_student.result())
            del tape
        ###################### testing with validation set ######################
        if i%check == 0:
            reconstructed_input = model(valid_data)
            valid_loss = tf.keras.losses.MeanSquaredError()(reconstructed_input, tf.abs(valid_data))
            np_data.log(i, [train_reconstruction_loss_student.result(), valid_loss])
            print("============================================================\n")
            print(valid_loss)
            if valid_loss < max_acc:
                max_acc = valid_loss
                model.save(fname_template.format(".h5"))
            if i >= (check * 2):
                graphing_data = np_data.data
                improvement = graphing_data[i + 1 - (check * 2): i - check + 1, -1].mean()*check - graphing_data[
                                                                                                   i - check + 1: i + 1,
                                                                                                   -1].mean()*check
                counter = 0
                for asldk in graphing_data[0:i + 1, -1]:
                    if asldk != 0:
                        print(counter, asldk)
                    counter = counter + 1
                print("the improvement in the past {} epochs is: ".format(check), improvement)
                print("the validation SR is: ", valid_loss)
                if improvement <= 0.0001:
                    break
        else:
            np_data.log(i, [train_reconstruction_loss_student.result(), 0])
    np_data.save()
    tf.keras.backend.clear_session()
def grid_search_multirate(bits = 8):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"

    fname_template_template = "trained_models/better_quantizer/multi_rate_{}"
    fname_template = fname_template_template.format(bits) + "{}"
    check = 250
    SUPERVISE_TIME = 0
    training_mode = 2
    swap_delay = check / 2

    # problem Definition
    M = 64
    K = 50
    B = 4
    E = 30
    more = bits
    seed = 100
    N_rf = 8
    sigma2_h = 6.3
    sigma2_n = 1.0
    res = 6
    ############################### generate data ###############################
    valid_data = generate_link_channel_data(1000, K, M, Nrf=N_rf)
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    q_valid_data = tf.abs(valid_data) / max_val
    q_valid_data = tf.where(q_valid_data > 1.0, 1.0, q_valid_data)
    q_valid_data = tf.round(q_valid_data * (2 ** res - 1)) / (2 ** res - 1) * max_val
    ################################ hyperparameters ###############################
    EPOCHS = 1000000
    lr = 0.001
    N = 400 # number of
    rounds = 1000
    sample_size = 100
    temp = 0.1
    check = 200
    model = CSI_reconstruction_model_seperate_decoders_multirate(M, K, B, E, N_rf, rates=more, avg_max=max_val)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    ################################ Metrics  ###############################
    train_reconstruction_loss_student = tf.keras.metrics.Mean(name='train_loss')
    ################################ storing train data in npy file  ##############################
    # the three would be first train_loss, Hardloss, and the validation loss, every 50 iterations
    max_acc = 10000
    np_data = ModelTrainer(save_dir=fname_template.format(".npy"), data_cols=2, epoch=EPOCHS)
    # training loop
    for i in range(0, EPOCHS):
        # generate training data
        train_reconstruction_loss_student.reset_states()
        train_data = generate_link_channel_data(N, K, M, Nrf=N_rf)
        ###################### training happens here ######################
        for e in range(0, 1):
            train_reconstruction_loss_student.reset_states()
            with tf.GradientTape(persistent=True) as tape:
                ###################### model post-processing ######################
                # q_train_data = tf.abs(train_data)/max_val
                # q_train_data = tf.where(q_train_data > 1.0, 1.0, q_train_data)
                # q_train_data = tf.round(q_train_data * (2 ** res - 1)) / (2 ** res - 1) * max_val
                reconstructed_input = model(train_data) # raw_ans is in the shape of (N, passes, M*K, N_rf)
                ###################### model post-processing ######################
                loss_student = tf.keras.losses.MeanSquaredError()(reconstructed_input, tf.abs(train_data))
            gradients_student = tape.gradient(loss_student, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients_student, model.trainable_variables))
            # optimizer.minimize(loss, ans)
            train_reconstruction_loss_student(loss_student)

            print(train_reconstruction_loss_student.result())
            del tape
        ###################### testing with validation set ######################
        if i%check == 0:
            reconstructed_input = model(valid_data)
            valid_loss = tf.keras.losses.MeanSquaredError()(reconstructed_input, tf.abs(valid_data))
            np_data.log(i, [train_reconstruction_loss_student.result(), valid_loss])
            print("============================================================\n")
            print(valid_loss)
            if valid_loss < max_acc:
                max_acc = valid_loss
                model.save(fname_template.format(".h5"))
            if i >= (check * 2):
                graphing_data = np_data.data
                improvement = graphing_data[i + 1 - (check * 2): i - check + 1, -1].mean()*check - graphing_data[
                                                                                                   i - check + 1: i + 1,
                                                                                                   -1].mean()*check
                counter = 0
                for asldk in graphing_data[0:i + 1, -1]:
                    if asldk != 0:
                        print(counter, asldk)
                    counter = counter + 1
                print("the improvement in the past {} epochs is: ".format(check), improvement)
                print("the validation SR is: ", valid_loss)
                if improvement <= 0.0001:
                    break
        else:
            np_data.log(i, [train_reconstruction_loss_student.result(), 0])
    np_data.save()
    tf.keras.backend.clear_session()
if __name__ == "__main__":
    # ranges = [[8,16,32,64],
    #  [8,16,32,32],
    #  [8,16,32],
    #  [8,16]]
    for N_rf_to_search in range(2, 129, 2):
        # grid_search_multirate(N_rf_to_search)
        # grid_search_VQVAE(N_rf_to_search)
        grid_search_STE(N_rf_to_search)

