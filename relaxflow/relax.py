import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from relaxflow.reparam import BinaryReparam, CategoricalReparam
import tqdm
import pdb
EPSILON = 1e-16


def killnan(X):
    return tf.where(tf.is_nan(X), tf.zeros_like(X), X)

def RELAX(tape, loss_fn, N, N_rf, K, M, gain_matrix, loss, control, conditional_control, logp,
          hard_params, params=[], var_params=[], weight=1.,
          handle_nan=False, summaries=False, report=False):
    '''Estimate the gradient of "loss" with respect to "hard_params" which
    enter the loss through a stochastic non-differentiable map.
    Use RELAX estimator for the gradient with respect to "hard_params" and
    use a standard sampling estimator for parameters in "params".

    The RELAX estimator estimates the gradient of f(b) where
    H(b)=z and z~p(z;hard_params) using a control variate function c().
    The input "control" corresponds to c(z) and "conditional_control" to c(zb)
    where zb~p(z|b). For a canonical construction, see "buildcontrol" function.
    Note: RELAX estimator may be wrong if construction is not correct.
    Respect above relationship between "control" and "conditional_control" and
    use tf.stop_gradient on the discrete variable as in "buildcontrol".
    Args:
        loss: scalar tensor to compute gradients for.
        control: differentiable tensor approximating loss.
        conditional_control: identical to control, but with noise
            conditioned on discrete sample.
        logp: a stochastic tensor equal to the log-probability ln p(b)
            of the discrete random variable b being sampled.
        params: other parameters. Uses same samples as REBAR.
        var_params: parameters that the estimator's variance
            should be minimized for.
        handle_nan: set NaNs occurring due to derivative operations to zero.
        split: Return

    Returns:
        gradients: REBAR grad ient estimator for params and hard_params.
        loss: function evaluated in discrete random sample.
        var_grad: gradient of estimator variance for var_params.
    '''
    with tf.name_scope("RELAX"):
        approx_gap = (loss - weight*conditional_control)
        blocked_approx_gap = approx_gap + tf.stop_gradient(approx_gap - approx_gap)
        with tf.name_scope("pure_grad"):
            # compute gradient of loss outside of dependence through b.
            temp_val = tf.reshape(hard_params[0], [N, N_rf, K * M])
            temp_val = tf.reduce_sum(temp_val, axis=1)
            pure_grads = tape.gradient([tf.reduce_mean(loss_fn(temp_val, gain_matrix))], hard_params)
        with tf.name_scope("score"):
            scores = tape.gradient([tf.reduce_mean(logp*blocked_approx_gap)], hard_params)
        # scores = tf.contrib.graph_editor.graph_replace(scores, {blocked_approx_gap: approx_gap})
        #pdb.set_trace()
        with tf.name_scope("relax_grad"):
            # Derivative of differentiable control variate.
            relax_grads = tape.gradient(tf.reduce_mean(control - conditional_control),
                                        hard_params)

        gradcollection = list(zip(pure_grads, relax_grads, hard_params, scores))
        hard_params_grads = []
        vectorized = []
        # var_params_grads = [(tf.zeros_like(var_param), var_param)
        #                    for var_param in var_params]
        if summaries:
            tf.summary.scalar('approx_gap', approx_gap)
        with tf.name_scope("collect_grads"):
            for pure_grad, relax_grad, hard_param, score in gradcollection:
                if handle_nan:
                    score = killnan(score)
                score_grad = score

                # aggregate gradient components
                full_grad = score_grad
                if pure_grad is not None:
                    if handle_nan:
                        pure_grad = killnan(pure_grad)
                    full_grad += pure_grad

                if relax_grad is not None:
                    if handle_nan:
                        relax_grad = killnan(relax_grad)
                    # complete RELAX estimator
                    full_grad += weight*relax_grad
                hard_params_grads += [(full_grad, hard_param)]
                vectorized.append(tf.reshape(full_grad, (-1,)))
            # with tf.name_scope("variance_grad"):
            #     grad_vec = tf.reduce_mean(tf.concat(vectorized, axis=0))
            #     grad_grads = tape.gradient(grad_vec, var_params)
            #     if handle_nan:
            #         grad_grads = [killnan(grad) for grad in grad_grads]
            #     var_params_grads = [(2*grad_vec*grad_grad, var_param)
            #                         for grad_grad, var_param in zip(grad_grads, var_params)]

        # with tf.name_scope("params_grad"):
        #     # ordinary parameter gradients
        #     params_grads = list(zip(tape.gradient(loss, params), params))
        return full_grad #+ [(weight_grad, weight)]
        # return params_grads

if __name__ is "__main__":

    np.random.seed(1)
    tf.set_random_seed(1)
    N = 20
    K = 10
    R = 10000  # sample repeats
    true_index = np.eye(K)[np.random.randint(0, K, N)]

    # Cost function with 1 good configuration
    def f(z):
        return -tf.identity(tf.reduce_sum(z) + tf.reduce_sum(z*true_index))

    for categorical in [False, True]:
        Z = tf.identity(tf.Variable(true_index, dtype=tf.float32))
        temp_var = tf.Variable(1.)
        temp = tf.nn.softplus(temp_var)
        nu_var = tf.Variable(1.)
        nu_switch = tf.Variable(1.)
        nu = nu_switch*(nu_var)
        var_summaries = [tf.summary.scalar("temp_" + ("cat" if categorical else "bin"), temp),
                          tf.summary.scalar("nu_" + ("cat" if categorical else "bin"), nu)]
        tempc_var = tf.Variable(1.)
        tempc = tf.nn.softplus(tempc_var)
        nuc_var = tf.Variable(1.)
        nuc_switch = tf.Variable(1.)
        nuc = nu_switch*(nuc_var)
        varc_summaries = [tf.summary.scalar("tempc_" + ("cat" if categorical else "bin"), tempc),
                          tf.summary.scalar("nuc_" + ("cat" if categorical else "bin"), nuc)]

        if categorical:
            rep = CategoricalReparam(Z, temperature=temp)
        else:
            rep = BinaryReparam(Z, temperature=temp)

        grad, var_grad = RELAX(*rep.rebar_params(f, weight=nu), [Z],
                               var_params=[temp_var, nu_var], handle_nan=True)

        if categorical:
            repc = CategoricalReparam(Z, coupled=True, temperature=tempc)
        else:
            repc = BinaryReparam(Z, coupled=True, temperature=tempc)

        gradc, varc_grad = RELAX(*repc.rebar_params(f, weight=nuc), [Z],
                                 var_params=[tempc_var, nuc_var], handle_nan=True)

        grad_estimator = tf.expand_dims(grad[0][0], 0)
        gradc_estimator = tf.expand_dims(gradc[0][0], 0)
        tf.train.AdamOptimizer()
        opt = tf.train.AdamOptimizer()
        train_step = opt.apply_gradients(var_grad)
        optc = tf.train.AdamOptimizer()
        trainc_step = opt.apply_gradients(varc_grad)

        train_writer = tf.summary.FileWriter('./opt_param')
        summaries = tf.summary.merge(var_summaries + varc_summaries)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        # Calculate gradients without REBAR
        sess.run(tf.assign(nu_switch, 0.))
        raw_grads = []
        print("Calculate score estimator")
        for _ in tqdm.tqdm(range(R), total=R):
            raw_grads += sess.run([grad_estimator])
        raw_grad = np.concatenate(raw_grads, axis=0)
        raw_mu = raw_grad.mean(axis=0)
        raw_var = np.square(raw_grad - raw_mu).mean(axis=0)

        #switch REBAR on
        sess.run(tf.assign(nu_switch, 1.))

        # Calculate gradients using REBAR
        base_grads = []
        print("Calculate REBAR estimator")
        for _ in tqdm.tqdm(range(R), total=R):
            base_grads += sess.run([(grad_estimator)])
        base_grad = np.concatenate(base_grads, axis=0)
        base_mu = base_grad.mean(axis=0)
        base_var = np.square(base_grad - base_mu).mean(axis=0)

        # Calculate gradients using REBAR with coupling
        couple_grads = []
        print("Calculate coupled estimator")
        for _ in tqdm.tqdm(range(R), total=R):
            couple_grads += sess.run([gradc_estimator])
        couple_grad = np.concatenate(couple_grads, axis=0)
        couple_mu = couple_grad.mean(axis=0)
        couple_var = np.square(couple_grad - couple_mu).mean(axis=0)

        # Optimize nu and temp, then Calculate gradients using REBAR
        Nsteps = 10000
        print("Optimize variance parameters")
        for step in tqdm.tqdm(range(Nsteps), total=Nsteps):
            _, _, notes = sess.run([train_step, trainc_step, summaries])
            train_writer.add_summary(notes, step)
        print("optimal temperature: {}".format(temp.eval()))
        print("optimal temperature (coupled): {}".format(tempc.eval()))
        print("optimal control weight: {}".format(nu.eval()))
        print("optimal control weight (coupled): {}".format(nuc.eval()))

        opt_grads = []
        print("Calculate optimized estimator")
        for _ in tqdm.tqdm(range(R), total=R):
            opt_grads += sess.run([grad_estimator])
        opt_grad = np.concatenate(opt_grads, axis=0)
        opt_mu = opt_grad.mean(axis=0)
        opt_var = np.square(opt_grad - opt_mu).mean(axis=0)

        optc_grads = []
        print("Calculate optimized coupled estimator")
        for _ in tqdm.tqdm(range(R), total=R):
            optc_grads += sess.run([gradc_estimator])
        optc_grad = np.concatenate(optc_grads, axis=0)
        optc_mu = optc_grad.mean(axis=0)
        optc_var = np.square(optc_grad - optc_mu).mean(axis=0)


        plt.subplot(1, 2, 2 if categorical else 1)
        svars = np.column_stack([raw_var.ravel(), base_var.ravel(),
                                 couple_var.ravel(), opt_var.ravel(),
                                 optc_var.ravel()])
        plt.boxplot(np.log(svars))
        plt.xticks(np.arange(1, 6), ['Score Estimator', 'REBAR',
                                     'REBAR (coupled)', 'Optimized REBAR',
                                     'optimized+coupled'])
        plt.ylabel("Log Sample Variance ({} samples)".format(R))
        plt.title("Categorical" if categorical else "Binary")
