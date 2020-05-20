import tensorflow as tf

from neuralprocesses import NeuralProcessParams, process
from neuralprocesses.loss import loglikelihood, kullback_leibler_gaussian


# @tf.function
def train_step(
    neurproc, optimizer, context_xs, context_ys, target_xs, target_ys, n_train_draws
):
    with tf.GradientTape() as tape:
        pred_ys_mu, z_mu, z_sigma, z_mu_ct, z_sigma_ct = neurproc(
            [context_xs], [context_ys], [target_xs], [target_ys], n_draws=n_train_draws
        )

        loglike_loss = tf.negative(loglikelihood(target_ys, pred_ys_mu))
        kl_loss = kullback_leibler_gaussian(z_mu, z_sigma, z_mu_ct, z_sigma_ct)
        loss = loglike_loss + kl_loss

    gradients = tape.gradient(loss, neurproc.trainable_variables)
    optimizer.apply_gradients(zip(gradients, neurproc.trainable_variables))

    return loss
