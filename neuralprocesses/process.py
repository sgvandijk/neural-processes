import tensorflow as tf
from tensorflow.keras.layers import Input

from neuralprocesses import NeuralProcessParams
from neuralprocesses.loss import loglikelihood, KLqp_gaussian
from neuralprocesses.network import xy_to_z_params, decoder_g


def neural_process(
    params: NeuralProcessParams, learning_rate=0.001, n_draws=7,
):
    """Set up complete, trainable neural process

    This will set up the full neural process, including encoder,
    decoder, loss function and training operator.

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
    params
        Neural process parameters
    learning_rate
        Base learning rate used in Adam optimizer
    n_draws
        Number of draws of Z per context set, or number of predictions made per target

    """
    context_xs = Input(params.dim_x)
    context_ys = Input(params.dim_y)

    target_xs = Input(params.dim_x)
    target_ys = Input(params.dim_y)

    # Concatenate context and target
    x_all = tf.concat([context_xs, target_xs], axis=0)
    y_all = tf.concat([context_ys, target_ys], axis=0)

    # Map input to z
    z_context = xy_to_z_params(context_xs, context_ys, params)
    z_target = xy_to_z_params(target_xs, target_ys, params)

    # z_all = xy_to_z_params(x_all, y_all, params)

    model = tf.keras.models.Model(
        inputs=[context_xs, context_ys, target_xs, target_ys],
        outputs=[z_context, z_target],
    )
    return model

    # Sample z
    epsilon = tf.random.normal([n_draws, params.dim_z])
    z_samples = tf.multiply(epsilon, z_all.sigma)
    z_samples = tf.add(z_samples, z_all.mu)

    # Map (z, x*) to y*
    y_pred_params = decoder_g(z_samples, target_xs, params)

    print([context_xs, context_ys, target_xs, target_ys])
    print(y_pred_params)

    # ELBO
    loglike = loglikelihood(target_ys, y_pred_params)
    KL_loss = KLqp_gaussian(z_all.mu, z_all.sigma, z_context.mu, z_context.sigma)
    loss = tf.negative(loglike) + KL_loss

    # optimisation
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    return train_op, loss
