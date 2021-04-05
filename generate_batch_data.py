import numpy as np
import random
import tensorflow as tf
import scipy as sp

def generate_link_channel_data(N, K, M, sigma2_h=0.1, sigma2_n=0.1):
    Lp = 1  # Number of Paths
    P = tf.constant(sp.linalg.dft(M), dtype=tf.complex64) # DFT matrix
    P = tf.expand_dims(P, 0)
    P = tf.tile(P, (N, 1, 1))
    LSF_UE = np.array([0.0, 0.0], dtype=np.float32)  # Mean of path gains
    Mainlobe_UE = np.array([0, 0], dtype=np.float32)  # Mean of the AoD range
    HalfBW_UE = np.array([30.0, 30.0], dtype=np.float32)  # Half of the AoD range
    h_act_batch = tf.constant(generate_batch_data(N, M, K, Lp, LSF_UE, Mainlobe_UE, HalfBW_UE), dtype=tf.complex64)
    # taking hermecian
    h_act_batch = tf.transpose(h_act_batch, perm=(0, 2, 1), conjugate=True)
    G = tf.matmul(h_act_batch, P)
    noise = tf.complex(tf.random.normal(G.shape, 0, sigma2_n, dtype=tf.float32),
                       tf.random.normal(G.shape, 0, sigma2_n, dtype=tf.float32))
    G_hat = G
    return G_hat

def generate_batch_data(batch_size,M,K,L,LSF_UE,Mainlobe_UE,HalfBW_UE):
    alphaR_input = np.zeros((batch_size,L,K))
    alphaI_input = np.zeros((batch_size,L,K))
    theta_input = np.zeros((batch_size,L,K))
    for kk in range(K):
        alphaR_input[:,:,kk] = np.random.normal(loc=LSF_UE[0], scale=1.0/np.sqrt(2), size=[batch_size,L])
        alphaI_input[:,:,kk] = np.random.normal(loc=LSF_UE[0], scale=1.0/np.sqrt(2), size=[batch_size,L])
        theta_input[:,:,kk] = np.random.uniform(low=Mainlobe_UE[0]-HalfBW_UE[0], high=Mainlobe_UE[0]+HalfBW_UE[0], size=[batch_size,L])
    #### Actual Channel
    from0toM = np.float32(np.arange(0, M, 1))
    alpha_act = alphaR_input + 1j*alphaI_input
    theta_act = (np.pi/180)*theta_input
    h_act = np.complex64(np.zeros((batch_size,M,K)))
    for kk in range(K):
        for ll in range(L):
            theta_act_expanded_temp = np.tile(np.reshape(theta_act[:,ll,kk],[-1,1]),(1,M))
            response_temp = np.exp(1j*np.pi*np.multiply(np.sin(theta_act_expanded_temp),from0toM))
            alpha_temp = np.reshape(alpha_act[:,ll,kk],[-1,1])
            h_act[:,:,kk] += (1/np.sqrt(L))*alpha_temp*response_temp
    return h_act
def generate_batch_data_with_angle(batch_size,M,K,L,LSF_UE,Mainlobe_UE,HalfBW_UE):
    alphaR_input = np.zeros((batch_size,L,K))
    alphaI_input = np.zeros((batch_size,L,K))
    theta_input = np.zeros((batch_size,L,K))
    for kk in range(K):
        alphaR_input[:,:,kk] = np.random.normal(loc=LSF_UE[0], scale=1.0/np.sqrt(2), size=[batch_size,L])
        alphaI_input[:,:,kk] = np.random.normal(loc=LSF_UE[0], scale=1.0/np.sqrt(2), size=[batch_size,L])
        theta_input[:,:,kk] = np.random.uniform(low=Mainlobe_UE[0]-HalfBW_UE[0], high=Mainlobe_UE[0]+HalfBW_UE[0], size=[batch_size,L])
    #### Actual Channel
    from0toM = np.float32(np.arange(0, M, 1))
    alpha_act = alphaR_input + 1j*alphaI_input
    theta_act = (np.pi/180)*theta_input
    h_act = np.complex64(np.zeros((batch_size,M,K)))
    for kk in range(K):
        for ll in range(L):
            theta_act_expanded_temp = np.tile(np.reshape(theta_act[:,ll,kk],[-1,1]),(1,M))
            response_temp = np.exp(1j*np.pi*np.multiply(np.sin(theta_act_expanded_temp),from0toM))
            alpha_temp = np.reshape(alpha_act[:,ll,kk],[-1,1])
            h_act[:,:,kk] += (1/np.sqrt(L))*alpha_temp*response_temp
    return (h_act, theta_act)

