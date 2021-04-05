import tensorflow as tf
from tf_agents.distributions import gumbel_softmax
from util import *
from models import *
import numpy as np
import scipy as sp
from keras_adabound.optimizers import AdaBound

custome_obj = {'Closest_embedding_layer': Closest_embedding_layer,
               'Interference_Input_modification': Interference_Input_modification,
               'Interference_Input_modification_no_loop': Interference_Input_modification_no_loop,
               "Interference_Input_modification_per_user": Interference_Input_modification_per_user,
               "Closest_embedding_layer_moving_avg": Closest_embedding_layer_moving_avg,
               "Per_link_Input_modification_more_G": Per_link_Input_modification_more_G,
               "Per_link_Input_modification_more_G_less_X": Per_link_Input_modification_more_G_less_X,
               "Per_link_Input_modification_even_more_G": Per_link_Input_modification_even_more_G,
               "Per_link_Input_modification_compress_XG": Per_link_Input_modification_compress_XG,
               "Per_link_Input_modification_compress_XG_alt": Per_link_Input_modification_compress_XG_alt,
               "Per_link_Input_modification_more_G_alt_2": Per_link_Input_modification_more_G_alt_2,
               "Per_link_Input_modification_compress_XG_alt_2": Per_link_Input_modification_compress_XG_alt_2,
               "Per_link_Input_modification_most_G": Per_link_Input_modification_most_G,
               "Per_link_sequential_modification": Per_link_sequential_modification,
               "Per_link_sequential_modification_compressedX": Per_link_sequential_modification_compressedX,
               "Per_link_Input_modification_most_G_raw_self": Per_link_Input_modification_most_G_raw_self,
               "Reduced_output_input_mod": Reduced_output_input_mod,
               "TopPrecoderPerUserInputMod": TopPrecoderPerUserInputMod,
               "X_extends": X_extends,
               "Per_link_Input_modification_most_G_col": Per_link_Input_modification_most_G_col,
               "Sparsemax": Sparsemax,
               "Sequential_Per_link_Input_modification_most_G_raw_self": Sequential_Per_link_Input_modification_most_G_raw_self,
               "Per_link_Input_modification_most_G_raw_self_sigmoid": Per_link_Input_modification_most_G_raw_self_sigmoid}
def grid_search_knowledge_distillation(more = 8):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template_template = "trained_models/better_quantizer_with_scheduling/student_teacher_{}bits_Nrf=8".format(more)
    fname_template = fname_template_template + "{}"
    check = 250
    SUPERVISE_TIME = 0
    training_mode = 2
    swap_delay = check / 2

    # problem Definition
    M = 64
    K = 50
    B = 4
    E = 30
    more = more
    seed = 100
    N_rf = 8
    sigma2_h = 6.3
    sigma2_n = 1.0
    res = 8
    ############################### generate data ###############################
    valid_data = generate_link_channel_data(1000, K, M, Nrf=N_rf)
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    q_valid_data = tf.abs(valid_data) / max_val
    q_valid_data = tf.where(q_valid_data > 1.0, 1.0, q_valid_data)
    q_valid_data = tf.round(q_valid_data * (2 ** res - 1)) / (2 ** res - 1) * max_val
    ################################ hyperparameters ###############################
    EPOCHS = 100000
    lr = 0.001
    N = 20 # number of
    rounds = 5
    sample_size = 100
    temp = 0.1
    check = 200
    # model = FDD_per_link_archetecture_more_G(M, K, 7, N_rf, True, max_val)
    model = Feedbakk_FDD_model_scheduler_knowledge_distillation(M, K, B, E, N_rf, 4, more=more, avg_max=max_val)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    ################################ Metrics  ###############################
    sum_rate = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    train_sum_rate = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_hard_loss = tf.keras.metrics.Mean(name='train_loss')
    train_loss_teacher = tf.keras.metrics.Mean(name='train_loss')
    train_hard_loss_teacher = tf.keras.metrics.Mean(name='train_loss')
    ################################ storing train data in npy file  ##############################
    # the three would be first train_loss, Hardloss, and the validation loss, every 50 iterations
    max_acc = 0
    np_data = ModelTrainer(save_dir=fname_template.format(".npy"), data_cols=5, epoch=EPOCHS)
    # training loop
    for i in range(0, EPOCHS):
        train_hard_loss.reset_states()
        # generate training data
        train_data = generate_link_channel_data(N, K, M, Nrf=N_rf)
        ###################### training happens here ######################
        for e in range(0, rounds):
            temp = 0.5 * np.exp(-4.5 / rounds * e) + 0.1
            temp = np.float32(temp)
            train_hard_loss.reset_states()
            train_loss.reset_states()
            train_loss_teacher.reset_states()
            train_hard_loss_teacher.reset_states()
            with tf.GradientTape(persistent=True) as tape:
                ###################### model post-processing ######################
                # q_train_data = tf.abs(train_data)/max_val
                # q_train_data = tf.where(q_train_data > 1.0, 1.0, q_train_data)
                # q_train_data = tf.round(q_train_data * (2 ** res - 1)) / (2 ** res - 1) * max_val
                scheduled_output, raw_output, teacher_scheduled_output, teacher_raw_output, reconstructed_input, reconstructed_input_teacher = model(train_data) # raw_ans is in the shape of (N, passes, M*K, N_rf)
                raw_ans = tf.transpose(raw_output, [0, 1, 3, 2])
                teacher_ans = tf.transpose(teacher_raw_output, [0, 1, 3, 2])
                out_raw = tf.reshape(raw_ans[:,-1], [N * N_rf, K*M])
                teacher_out_raw = tf.reshape(teacher_ans[:, -1], [N * N_rf, K * M])
                sm = gumbel_softmax.GumbelSoftmax(temperature=temp, logits=out_raw)
                out_raw = sm.sample(sample_size)
                teacher_sm = gumbel_softmax.GumbelSoftmax(temperature=temp, logits=teacher_out_raw)
                teacher_out_raw = teacher_sm.sample(sample_size)
                out_raw = tf.reshape(out_raw, [sample_size, N, N_rf, K*M])
                teacher_out_raw = tf.reshape(teacher_out_raw, [sample_size, N, N_rf, K * M])
                out = tf.reduce_sum(out_raw, axis=2)
                out = tf.reshape(out, [sample_size*N, K*M])
                teacher_out = tf.reduce_sum(teacher_out_raw, axis=2)
                teacher_out = tf.reshape(teacher_out, [sample_size * N, K * M])
                train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [100,1, 1, 1]), [100*N, K, M])
                ###################### model post-processing ######################
                loss_student = train_sum_rate(out, train_label)
                loss_teacher = train_sum_rate(teacher_out, train_label)
                loss = loss_student + loss_teacher
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,model.trainable_variables))
            # optimizer.minimize(loss, ans)
            train_loss(loss_student)
            train_loss_teacher(loss_teacher)
            train_hard_loss(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(scheduled_output[:,-1]), train_data))
            train_hard_loss_teacher(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(teacher_scheduled_output[:, -1]), train_data))
            print(train_loss_teacher.result(), train_hard_loss_teacher.result(), train_hard_loss.result(),train_loss.result())
            del tape
        ###################### testing with validation set ######################
        if i%check == 0:
            scheduled_output, raw_output, teacher_scheduled_output, teacher_raw_output, reconstructed_input, reconstructed_input_teacher  = model.predict(q_valid_data, batch_size=N)
            valid_loss = tf.reduce_mean(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(scheduled_output[:, -1]), valid_data))
            np_data.log(i, [train_loss_teacher.result(), train_hard_loss_teacher.result(), train_hard_loss.result(), train_loss.result(), valid_loss])
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
                print("the improvement in the past 500 epochs is: ", improvement)
                print("the validation SR is: ", valid_loss)
                if improvement <= 0.0001:
                    break
        else:
            np_data.log(i, [train_loss_teacher.result(), train_hard_loss_teacher.result(), train_hard_loss.result(), train_loss.result(), 0])
    np_data.save()
def grid_search_STD(more = 8):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template_template = "trained_models/Dec28/NRF=5/shifted/GNN_annealing_temp_B={}+limit_res=4".format(more)
    fname_template = fname_template_template + "{}"
    check = 250
    SUPERVISE_TIME = 0
    training_mode = 2
    swap_delay = check / 2

    # problem Definition
    M = 64
    K = 50
    B = 4
    E = 30
    more = more
    seed = 100
    N_rf = 5
    sigma2_h = 6.3
    sigma2_n = 1.0
    res = 4
    ############################### generate data ###############################
    valid_data = generate_link_channel_data(1000, K, M, Nrf=N_rf)
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    q_valid_data = tf.abs(valid_data) / max_val
    q_valid_data = tf.where(q_valid_data > 1.0, 1.0, q_valid_data)
    capped_valid = q_valid_data
    q_valid_data = (tf.round(q_valid_data * (2 ** res - 1)) / (2 ** res - 1)) * max_val
    ################################ hyperparameters ###############################
    EPOCHS = 100000
    lr = 0.001
    N = 40 # number of
    rounds = 5
    sample_size = 100
    temp = 0.1
    check = 200
    # model = FDD_per_link_archetecture_more_G(M, K, 7, N_rf, True, max_val)
    model = Feedbakk_FDD_model_scheduler_naive(M, K, B, E, N_rf, 4, more=more, avg_max=max_val)
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
        train_data = generate_link_channel_data(N, K, M, Nrf=N_rf)
        ###################### training happens here ######################
        for e in range(0, rounds):
            temp = 0.5 * np.exp(-4.5 / rounds * e) + 0.1
            temp = np.float32(temp)
            train_hard_loss.reset_states()
            train_loss.reset_states()
            with tf.GradientTape(persistent=True) as tape:
                ###################### model post-processing ######################
                q_train_data = tf.abs(train_data)/max_val
                q_train_data = tf.where(q_train_data > 1.0, 1.0, q_train_data)
                q_train_data = (tf.round(q_train_data * (2 ** res - 1)) / (2 ** res - 1))* max_val
                scheduled_output, raw_output, reconstructed_input = model(q_train_data) # raw_ans is in the shape of (N, passes, M*K, N_rf)
                raw_ans = tf.transpose(raw_output, [0, 1, 3, 2])
                out_raw = tf.reshape(raw_ans[:,-1], [N * N_rf, K*M])
                sm = gumbel_softmax.GumbelSoftmax(temperature=temp, logits=out_raw)
                out_raw = sm.sample(sample_size)
                out_raw = tf.reshape(out_raw, [sample_size, N, N_rf, K*M])
                out = tf.reduce_sum(out_raw, axis=2)
                out = tf.reshape(out, [sample_size*N, K*M])
                train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [100,1, 1, 1]), [100*N, K, M])
                ###################### model post-processing ######################
                loss = train_sum_rate(out, train_label) + tf.keras.losses.MeanSquaredError()(reconstructed_input, tf.abs(q_train_data))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            # optimizer.minimize(loss, ans)
            train_loss(loss)
            train_hard_loss(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(scheduled_output[:,-1]), train_data))
            print(train_hard_loss.result(),train_loss.result())
            del tape
        ###################### testing with validation set ######################
        if i%check == 0:
            scheduled_output, raw_output, reconstructed_input = model.predict(q_valid_data, batch_size=N)
            valid_loss = tf.reduce_mean(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(scheduled_output[:, -1]), valid_data))
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), valid_loss])
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
                print("the improvement in the past 500 epochs is: ", improvement)
                print("the validation SR is: ", valid_loss)
                if improvement <= 0.0001:
                    break
        else:
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), 0])
    np_data.save()
    tf.keras.backend.clear_session()
def grid_search_pretrained(bits):
    path = "trained_models/better_quantizer/ste_models/STE_{}bits.h5".format(bits)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template_template = "trained_models/Dec28/NRF=5/pretrained_encoder/GNN_annealing_temp_B={}+limit_res=6".format(bits)
    fname_template = fname_template_template + "{}"
    check = 250
    SUPERVISE_TIME = 0
    training_mode = 2
    swap_delay = check / 2

    # problem Definition
    M = 64
    K = 50
    B = 4
    E = 30
    seed = 100
    N_rf = 5
    sigma2_h = 6.3
    sigma2_n = 1.0
    res = 6
    ############################### generate data ###############################
    valid_data = generate_link_channel_data(1000, K, M, Nrf=N_rf)
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    q_valid_data = tf.abs(valid_data) / max_val
    q_valid_data = tf.where(q_valid_data > 1.0, 1.0, q_valid_data)
    q_valid_data = (tf.round(q_valid_data * (2 ** res - 1)) / (2 ** res - 1)  + 1/2**(res+1)) * max_val
    ################################ hyperparameters ###############################
    EPOCHS = 100000
    lr = 0.001
    N = 40 # number of
    rounds = 5
    sample_size = 100
    temp = 0.1
    check = 200
    # model = FDD_per_link_archetecture_more_G(M, K, 7, N_rf, True, max_val)
    model = Feedbakk_FDD_model_scheduler_pretrained(M, K, B, E, N_rf, 4, path, custome_obj, avg_max=max_val)
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
        train_data = generate_link_channel_data(N, K, M, Nrf=N_rf)
        ###################### training happens here ######################
        for e in range(0, rounds):
            temp = 0.5 * np.exp(-4.5 / rounds * e) + 0.1
            temp = np.float32(temp)
            train_hard_loss.reset_states()
            train_loss.reset_states()
            with tf.GradientTape(persistent=True) as tape:
                ###################### model post-processing ######################
                q_train_data = tf.abs(train_data)/max_val
                q_train_data = tf.where(q_train_data > 1.0, 1.0, q_train_data)
                q_train_data = (tf.round(q_train_data * (2 ** res - 1)) / (2 ** res - 1) + 1/(2**(res+1)))* max_val
                scheduled_output, raw_output, reconstructed_input = model(q_train_data) # raw_ans is in the shape of (N, passes, M*K, N_rf)
                raw_ans = tf.transpose(raw_output, [0, 1, 3, 2])
                out_raw = tf.reshape(raw_ans[:,-1], [N * N_rf, K*M])
                sm = gumbel_softmax.GumbelSoftmax(temperature=temp, logits=out_raw)
                out_raw = sm.sample(sample_size)
                out_raw = tf.reshape(out_raw, [sample_size, N, N_rf, K*M])
                out = tf.reduce_sum(out_raw, axis=2)
                out = tf.reshape(out, [sample_size*N, K*M])
                train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [100,1, 1, 1]), [100*N, K, M])
                ###################### model post-processing ######################
                loss = train_sum_rate(out, train_label) + tf.keras.losses.MeanSquaredError()(reconstructed_input, tf.abs(q_train_data))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            # optimizer.minimize(loss, ans)
            train_loss(loss)
            train_hard_loss(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(scheduled_output[:,-1]), train_data))
            print(train_hard_loss.result(),train_loss.result())
            del tape
        ###################### testing with validation set ######################
        if i%check == 0:
            scheduled_output, raw_output, reconstructed_input = model.predict(q_valid_data, batch_size=N)
            valid_loss = tf.reduce_mean(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(scheduled_output[:, -1]), valid_data))
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), valid_loss])
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
                print("the improvement in the past 500 epochs is: ", improvement)
                print("the validation SR is: ", valid_loss)
                if improvement <= 0.0001:
                    break
        else:
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), 0])
    np_data.save()
    tf.keras.backend.clear_session()

if __name__ == "__main__":
    # for N_rf_to_search in range(1,25,2):
    #     grid_search_STD(N_rf_to_search)
    # for N_rf_to_search in range(65,129,2):
    for bits in range(34, 130, 2):
    # for bits in [128,64,32,16]:
        grid_search_pretrained(bits)
