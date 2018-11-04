import tensorflow as tf

from neuralprocesses.loss import loglikelihood, KLqp_gaussian
from neuralprocesses.network import xy_to_z_params, decoder_g


def init_neural_process(context_xs: tf.Tensor, context_ys: tf.Tensor,
                        target_xs: tf.Tensor, target_ys: tf.Tensor,
                        dim_r: int, dim_z: int,
                        n_hidden_units_h: int, n_hidden_units_g: int,
                        learning_rate=0.001, n_draws=7):
    """Set up complete, trainable neural process

    This will set up the full neaural process, including encoder, decoder, loss function and training operator. Any
    calls to network setup or prediction functions will reuse (parts of) the operations created here.

    Parameters
    ----------
    context_xs
        (Placeholder) tensor of the features in the context set, shape (n_samples, dim_x)
    context_ys
        (Placeholder) tensor of the outputs in the context set, shape (n_samples, dim_y)
    target_xs
        (Placeholder) tensor of the features in the target set, shape (n_target, dim_x)
    target_ys
        (Placeholder) tensor of the outputs in the target set, shape (n_targets, dim_y)
    dim_r
        Number of encoding dimensions
    dim_z
        Number of dimensions of Z
    n_hidden_units_h
        Number of hidden units for encoder network
    n_hidden_units_g
        Number of hidden units for the decoder netowrk
    learning_rate
        Base learning rate used in Adam optimizer
    n_draws
        Number of draws of Z per context set, or number of predictions made per target

    """
    # Concatenate context and target
    x_all = tf.concat([context_xs, target_xs], axis=0)
    y_all = tf.concat([context_ys, target_ys], axis=0)

    # Map input to z
    z_context = xy_to_z_params(context_xs, context_ys, n_hidden_units_h, dim_r, dim_z)
    z_all = xy_to_z_params(x_all, y_all, n_hidden_units_h, dim_r, dim_z)

    # Sample z
    epsilon = tf.random_normal([n_draws, dim_z])
    z_samples = tf.multiply(epsilon, z_all.sigma)
    z_samples = tf.add(z_samples, z_all.mu)

    # Map (z, x*) to y*
    y_pred_params = decoder_g(z_samples, target_xs, n_hidden_units_g)

    # ELBO
    loglike = loglikelihood(target_ys, y_pred_params)
    KL_loss = KLqp_gaussian(z_all.mu, z_all.sigma, z_context.mu, z_context.sigma)
    loss = tf.negative(loglike) + KL_loss

    # optimisation
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    return train_op, loss
