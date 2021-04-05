import tensorflow as tf
from tf_agents.distributions import gumbel_softmax
from util import *
from models import *
import numpy as np
import scipy as sp
from keras_adabound.optimizers import AdaBound
# from matplotlib import pyplot as plt
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
               "Reduced_output_input_mod":Reduced_output_input_mod,
               "TopPrecoderPerUserInputMod":TopPrecoderPerUserInputMod,
               "X_extends": X_extends}
# from matplotlib import pyplot as plt
def train_prev_outs(scheduled_output, features):
    for i in range(0, scheduled_output.shape[1]):
        sr = sum_rate(scheduled_output[:, i], features)
        loss_1 = loss_1 + tf.exp(tf.constant(-scheduled_output.shape[1] + 1 + i, dtype=tf.float32)) * sr
        # ce = All_softmaxes_CE_general(N_rf, K, M)(raw_output[:, i])
        # loss_4 = loss_4 + factor[N_rf] * tf.exp(tf.constant(-scheduled_output.shape[1]+1+i, dtype=tf.float32)) * ce
        mask = tf.stop_gradient(Harden_scheduling(k=N_rf)(scheduled_output[:, i]))
        ce = tf.keras.losses.CategoricalCrossentropy()(scheduled_output[:, i], mask)
        loss_4 = loss_4 + 0.1 * tf.exp(tf.constant(-scheduled_output.shape[1] + 1 + i, dtype=tf.float32)) * ce
def train_step(features, labels, N=None, epoch=0, lr_boost=1.0, reg_strength = 1.0):
    with tf.GradientTape(persistent=True) as tape:
        # compressed_G, position_matrix = G_compress(features, 2)
        # scheduled_output, raw_output = model([features, compressed_G, position_matrix])
        limited_features = feedback_model(features)

        scheduled_output, raw_output = model(limited_features)
        # mask = tf.stop_gradient(Harden_scheduling(k=N_rf)(overall_softmax))
        # scheduled_output, raw_output, z_qq, z_e, reconstructed_input = model(features)
        # loss_1 = tf.keras.losses.MeanSquaredError()(reconstructed_input, tf.abs(features))
        # loss_1 = tf.reduce_mean(sum_rate_train(scheduled_output[:, -1], features))
        pred = scheduled_output[:, -1]
        loss = tf.reduce_mean(sum_rate_train(pred, limited_features))
        mask = tf.stop_gradient(Harden_scheduling_user_constrained(N_rf, K, M, default_val=0)(scheduled_output[:, -1]))
        ce = tf.keras.losses.CategoricalCrossentropy()(scheduled_output[:, -1], mask)
        loss = loss + 0.1 * ce

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # gradients = tape.gradient(loss, model.get_layer("model").trainable_variables)
    # optimizer2.apply_gradients(zip(gradients, model.get_layer("model").trainable_variables))
    # gradients_2 = tape.gradient(loss_4, model.get_layer("scheduler").trainable_variables)
    # optimizer.apply_gradients(zip(gradients_2, model.get_layer("scheduler").trainable_variables))

    train_loss(loss)
    train_hard_loss(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(pred), features))
    try:
        train_binarization_loss(loss_3)
    except:
        timmer = 0
    del tape
    return train_hard_loss.result()
if __name__ == "__main__":
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template = "trained_models/Nov_22/Nrf={}links={}bits={}no_vae{}"
    check = 250
    SUPERVISE_TIME = 0
    training_mode = 2
    swap_delay = check / 2

    # problem Definition
    N = 50
    M = 64
    K = 50
    B = 4
    E = 30
    more = 64
    seed = 100
    N_rf = 8
    sigma2_h = 6.3
    sigma2_n = 1.0
    # hyperparameters
    EPOCHS = 100000
    # EPOCHS = 1
    valid_data = generate_link_channel_data(1000, K, M, Nrf=N_rf)
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    model = FDD_per_link_archetecture_more_G(M, K, 5, N_rf, True, max_val)
    for i in range(0, 1000):
        train_data = generate_link_channel_data(N, K, M, Nrf=N_rf)
        sum_rate = Sum_rate_utility_WeiCui(K, M, 1)
        train_sum_rate = Sum_rate_utility_WeiCui(K, M, 0)
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_hard_loss = tf.keras.metrics.Mean(name='train_loss')
        optimizer = tf.keras.optimizers.Adam(lr=0.001)
        train_hard_loss.reset_states()
        train_hard_loss(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(out), train_label))
        for e in range(0, 200):
            train_loss.reset_states()
            with tf.GradientTape(persistent=True) as tape:
                ans, raw_ans = model(train_data)
                raw_ans = tf.transpose(raw_ans, [0, 1, 3, 2])
                ###################### shape flatten ######################
                out_raw = tf.reshape(raw_ans[:,-1], [N * N_rf, K*M])
                temp = 0.1
                sm = gumbel_softmax.GumbelSoftmax(temperature=temp, logits=out_raw)
                out_raw = sm.sample(100)

                # out_user = tf.keras.layers.Softmax(axis=1)(ans_user)
                # out_user = tf.reduce_sum(out_user, axis=2, keepdims=True)
                # out_link = tf.keras.layers.Softmax(axis=2)(ans_link)
                # out = tf.multiply(out_user, out_link)
                out_raw = tf.reshape(out_raw, [100, N, N_rf, K*M])
                out = tf.reduce_sum(out_raw, axis=2)
                out = tf.reshape(out, [100*N, K*M])
                train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [100,1, 1, 1]), [100*N, K, M])
                loss = train_sum_rate(out, train_label)
                # ce = All_softmaxes_MSE_general(N_rf, K, M)(out_raw)
                loss = loss
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,model.trainable_variables))
            # optimizer.minimize(loss, ans)
            train_loss(loss)
            train_hard_loss(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(out), train_label))

            print(train_hard_loss.result(),train_loss.result())
            del tape
        print("====================\n", train_hard_loss.result(), train_loss.result())
        from matplotlib import pyplot as plt
        # plt.plot(out[0].numpy())
        # plt.show()
    A[2]


    mores = [2, 1]
    Es = [8, 7, 6, 5, 4, 3, 2, 1]
    for j in Es:
        for i in mores:
            N_rf = j
            bits = 8
            links = i
            tf.random.set_seed(1)
            np.random.seed(1)
            valid_data = generate_link_channel_data(1000, K, M, Nrf=N_rf)

            garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
            mean_val = tf.reduce_mean(tf.abs(valid_data))
            # ==================== hieristic feedback ====================
            # feedback_model = k_link_feedback_model(N_rf, bits, links, M, K, max_val)
            # valid_data_in = feedback_model(valid_data)
            # ==================== hieristic feedback ====================
            reg_strength = 1.0
            model = FDD_per_link_archetecture_more_G(M, K, 12, N_rf, True, max_val)
            # print(model.summary())
            lambda_var_1 = tf.Variable(1.0, trainable=True)
            lambda_var_2 = tf.Variable(1.0, trainable=True)
            lambda_var_3 = tf.Variable(1.0, trainable=True)

            # model = FDD_RNN_model(M, K, N_rf)

            # model = FDD_per_link_2Fold(M, K, 6, N_rf, output_all=True)
            # model = Top2Precoder_model(M, K, 4, N_rf, 2)
            # model = CSI_reconstruction_model_seperate_decoders_input_mod(M, K, 6, N_rf, output_all=True, more=more)
            # model = FDD_reduced_output_space(M, K, N_rf)
            # model = FDD_distributed_then_general_architecture(M, K, k=2, N_rf=N_rf, output_all=False)
            # model = Feedbakk_FDD_mcodel_scheduler(M, K, B, E, N_rf, 6, more=more, qbit=0, output_all=True)
            # model = Feedbakk_FDD_model_scheduler_naive(M, K, B, E, N_rf, 6, more=more, qbit=0, output_all=True)
            vae_loss = VAE_loss_general(False)
            sum_rate = Sum_rate_utility_WeiCui(K, M, sigma2_n)
            sum_rate_hard = Sum_rate_utility_hard(K, M, sigma2_n)
            sum_rate_train = Sum_rate_utility_WeiCui(K, M, sigma2_n)
            sum_rate_interference = Sum_rate_interference(K, M, sigma2_n)

            optimizer = tf.keras.optimizers.Adam(lr=0.01)
            optimizer2 = tf.keras.optimizers.Adam(lr=0.01)
            # optimizer = tf.keras.optimizers.SGD(lr=0.001)
            # for data visualization
            graphing_data = np.zeros((EPOCHS, 4))
            train_loss = tf.keras.metrics.Mean(name='train_loss')
            train_binarization_loss = tf.keras.metrics.Mean(name='train_loss')
            train_hard_loss = tf.keras.metrics.Mean(name='train_loss')
            valid_sum_rate = tf.keras.metrics.Mean(name='valid_loss')
            # begin setting up training loop
            max_acc = 10000
            max_acc_loss = 10000
            # training Loop

            for epoch in range(EPOCHS):
                # ======== ======== data recording features ======== ========
                train_loss.reset_states()
                train_binarization_loss.reset_states()
                train_hard_loss.reset_states()
                valid_sum_rate.reset_states()
                # ======== ======== training step ======== ========
                # if epoch % 20 == 0:
                train_features = generate_link_channel_data(N, K, M, N_rf)
                current_result = train_step(train_features, None, training_mode, epoch=epoch, reg_strength=reg_strength)
                template = 'Epoch {}, Loss: {}, reconstruction_loss:{}, Hard Loss: {}'
                print(template.format(epoch,
                                      train_loss.result(),
                                      train_binarization_loss.result(),
                                      train_hard_loss.result()))
                graphing_data[epoch, 0] = train_loss.result()
                graphing_data[epoch, 1] = train_binarization_loss.result()
                graphing_data[epoch, 3] = train_hard_loss.result()
                # if train_hard_loss.result() < max_acc_loss:
                #     max_acc_loss = train_hard_loss.result()
                #     model.save(fname_template.format(i, "_max_train2.h5"))
                #     tim = tf.keras.models.load_model(fname_template.format(i, "_max_train2.h5"), custom_objects=custome_obj)

                if epoch % check == 0:
                    # compressed_G, position_matrix = G_compress(valid_data, 2)
                    # scheduled_output, raw_output = model.predict_on_batch([valid_data, compressed_G, position_matrix])
                    # scheduled_output, raw_output = model.predict(valid_data, batch_size=N)
                    scheduled_output, raw_output = model.predict(valid_data, batch_size=N)
                    pred = scheduled_output[:, -1]
                    # scheduled_output, raw_output, z_qq, z_e, reconstructed_input = model.predict(valid_data, batch_size=N)
                    out = sum_rate(Harden_scheduling_user_constrained(N_rf, K, M, default_val=0)(pred), tf.abs(valid_data))
                    # out = tf.keras.losses.MeanSquaredError()(reconstructed_input, tf.abs(valid_data)/max_val) # with vqvae
                    valid_sum_rate(out)
                    graphing_data[epoch, 2] = valid_sum_rate.result()
                    if valid_sum_rate.result() < max_acc:
                        max_acc = valid_sum_rate.result()
                        model.save(fname_template.format(N_rf, links, bits, ".h5"))
                    if epoch >= (SUPERVISE_TIME) and epoch >= (check * 2):
                        improvement = graphing_data[epoch + 1 - (check * 2): epoch - check + 1, 2].min() - graphing_data[
                                                                                                    epoch - check + 1: epoch + 1,
                                                                                                    2].min()
                        counter = 0
                        for asldk in graphing_data[0:epoch+1, 2]:
                            if asldk != 0:
                                print(counter, asldk)
                            counter = counter + 1
                        print("the improvement in the past 500 epochs is: ", improvement)
                        print("the validation SR is: ", valid_sum_rate.result())
                        if improvement <= 0.0001:
                            break
            np.save(fname_template.format(N_rf, links, bits, ".npy"), graphing_data)
            tf.keras.backend.clear_session()
            print("Training end")


