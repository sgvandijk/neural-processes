import tensorflow as tf
from tensorflow.keras.layers import Input

from neuralprocesses import NeuralProcessParams, loss, network


class NeuralProcess(tf.keras.Model):
    def __init__(self, params: NeuralProcessParams):
        super(NeuralProcess, self).__init__(self)
        self.params = params

        self.encoder = network.Encoder(params)
        self.decoder = network.Decoder(params)

    def call(self, context_xs, context_ys, target_xs, target_ys=None, n_draws=7):
        batch_size = tf.shape(context_xs)[0]

        # Encode context to latent distribution z
        z_mu, z_sigma = self.encoder(context_xs, context_ys)

        # Sample z
        epsilon = tf.random.normal([batch_size, n_draws, self.params.dim_z])
        z_samples = tf.multiply(epsilon, z_sigma)
        z_samples = tf.add(z_samples, z_mu)

        # Map (z, x*) to y*
        pred_ys_mu = self.decoder(target_xs, z_samples)

        if target_ys is None:
            # We're done, return the predicted means, as well as the latent distribution
            return pred_ys_mu, z_mu, z_sigma
        else:
            # Looks like we are training, so also encode full set including target_xs
            context_target_xs = tf.concat([context_xs, target_xs], axis=-2)
            context_target_ys = tf.concat([context_ys, target_ys], axis=-2)
            z_mu_ct, z_sigma_ct = self.encoder(context_target_xs, context_target_ys)
            return pred_ys_mu, z_mu, z_sigma, z_mu_ct, z_sigma_ct

    # # ELBO
    # loglike = loglikelihood(target_ys, y_pred_params)
    # KL_loss = KLqp_gaussian(z_all.mu, z_all.sigma, z_context.mu, z_context.sigma)
    # loss = tf.negative(loglike) + KL_loss

    # # optimisation
    # optimizer = tf.train.AdamOptimizer(learning_rate)
    # train_op = optimizer.minimize(loss)

    # return train_op, loss
