from typing import Optional, Union

import numpy as np
import tensorflow as tf

from neuralprocesses import NeuralProcessParams


def prior_predict(
    input_xs: Union[np.array, tf.Tensor],
    decoder: tf.keras.models.Model,
    epsilon: Optional[tf.Tensor] = None,
    n_draws: int = 1,
) -> tf.Tensor:
    """Predict output with random network

    This can be seen as a prior over functions, where no training
    and/or context data is seen yet. The decoder g is randomly
    initialised, and random samples of Z are drawn from a standard
    normal distribution, or taken from `epsilon` if provided.

    Parameters
    ----------
    input_xs_value
        Values of input features to predict for, shape: (n_samples, dim_x)
    params
        Neural process parameters
    epsilon
        Optional samples for Z. If omitted, samples will be drawn from a standard normal distribution.
        Shape: (n_draws, dim_z)
    n_draws
        Number of samples for Z to draw if `epsilon` is omitted

    Returns
    -------
        Output tensors for the parameters of Gaussian distributions for y*

    """
    # the source of randomness can be optionally passed as an argument
    if epsilon is None:
        epsilon = tf.random.normal([n_draws, params.dim_z])
    z_sample = epsilon

    # Turn into single batch
    input_xs = tf.expand_dims(input_xs, 0)
    z_sample = tf.expand_dims(z_sample, 0)

    y_stars = decoder(input_xs, z_sample)
    return tf.squeeze(y_stars)


def posterior_predict(
    context_xs,
    context_ys,
    input_xs,
    neurproc,
    epsilon: Optional[tf.Tensor] = None,
    n_draws: int = 1,
):
    """Predict posterior function value conditioned on context

    Parameters
    ----------
    context_xs_value
        Array of context input values; shape: (n_samples, dim_x)
    context_ys_value
        Array of context output values; shape: (n_samples, dim_x)
    input_xs_value
        Array of input values to predict for, shape: (n_targets, dim_x)
    params
        Neural process parameters
    epsilon
        Source of randomness for drawing samples from latent variable
    n_draws
        How many samples to draw from latent variable; ignored if epsilon is given

    Returns
    -------
        Output tensors for the parameters of Gaussian distributions for y*
    """
    if epsilon is None:
        epsilon = tf.random.normal([n_draws, params.dim_z])
    z_sample = epsilon

    # Turn into single batch
    input_xs = tf.expand_dims(input_xs, 0)
    z_sample = tf.expand_dims(z_sample, 0)

    z_samples = tf.multiply(epsilon, z_params.sigma)
    z_samples = tf.add(z_samples, z_params.mu)

    y_star = decoder_g(z_samples, x_star, params)

    return y_star
