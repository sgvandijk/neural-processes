from typing import Optional

import numpy as np
import tensorflow as tf

from neuralprocesses import GaussianParams
from neuralprocesses.network import decoder_g, xy_to_z_params


def prior_predict(input_xs_value: np.array, dim_z: int, n_hidden_units_g: int,
                  epsilon: Optional[tf.Tensor] = None, n_draws: int = 1) -> GaussianParams:
    """Predict output with random network

    This can be seen as a prior over functions, where no training and/or context data is seen yet. The decoder g is
    randomly initialised, and random samples of Z are drawn from a standard normal distribution, or taken from
    `epsilon` if provided.

    Parameters
    ----------
    input_xs_value
        Values of input features to predict for, shape: (n_samples, dim_x)
    dim_z
        Number of dimensions of Z
    n_hidden_units_g
        Number of hidden units used in the decoder
    epsilon
        Optional samples for Z. If omitted, samples will be drawn from a standard normal distribution.
        Shape: (n_draws, dim_z)
    n_draws
        Number of samples for Z to draw if `epsilon` is omitted

    Returns
    -------
        Output tensors for the parameters of Gaussian distributions for y*
    """
    x_star = tf.constant(input_xs_value, dtype=tf.float32)

    # the source of randomness can be optionally passed as an argument
    if epsilon is None:
        epsilon = tf.random_normal((n_draws, dim_z))
    z_sample = epsilon

    y_star = decoder_g(z_sample, x_star, n_hidden_units_g)
    return y_star


def posterior_predict(context_xs_value: np.array, context_ys_value: np.array, input_xs_value: np.array,
                      dim_r: int, dim_z: int, n_hidden_units_h: int, n_hidden_units_g: int,
                      epsilon: Optional[tf.Tensor] = None, n_draws: int = 1) -> GaussianParams:
    """Predict posterior function value conditioned on context

    Parameters
    ----------
    context_xs_value
        Array of context input values; shape: (n_samples, dim_x)
    context_ys_value
        Array of context output values; shape: (n_samples, dim_x)
    input_xs_value
        Array of input values to predict for, shape: (n_targets, dim_x)
    dim_r
        Dimension of encoded representation of context
    dim_z
        Dimension of latent variable
    n_hidden_units_h
        Number of hidden units used in encoder
    n_hidden_units_g
        Number of hidden units used in decoder
    epsilon
        Source of randomness for drawing samples from latent variable
    n_draws
        How many samples to draw from latent variable; ignored if epsilon is given

    Returns
    -------
        Output tensors for the parameters of Gaussian distributions for y*
    """

    # Inputs for prediction time
    xs = tf.constant(context_xs_value, dtype=tf.float32)
    ys = tf.constant(context_ys_value, dtype=tf.float32)
    x_star = tf.constant(input_xs_value, dtype=tf.float32)

    # For out-of-sample new points
    z_params = xy_to_z_params(xs, ys, n_hidden_units_h, dim_r, dim_z)

    # the source of randomness can be optionally passed as an argument
    if epsilon is None:
        epsilon = tf.random_normal((n_draws, dim_z))
    z_samples = tf.multiply(epsilon, z_params.sigma)
    z_samples = tf.add(z_samples, z_params.mu)

    y_star = decoder_g(z_samples, x_star, n_hidden_units_g)

    return y_star
