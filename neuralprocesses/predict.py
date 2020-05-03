from typing import Optional, Union

import numpy as np
import tensorflow as tf

from neuralprocesses import GaussianParams, NeuralProcessParams
from neuralprocesses.network import decoder_g, xy_to_z_params


@tf.function
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

    y_stars = tf.TensorArray(dtype=tf.float32, size=n_draws)
    for i_z in range(n_draws):
        y_stars = y_stars.write(
            i_z,
            decoder([tf.expand_dims(input_xs, 0), tf.expand_dims(z_sample[i_z], 0)]),
        )

    return tf.squeeze(y_stars.stack())


def posterior_predict(
    context_xs_value: np.array,
    context_ys_value: np.array,
    input_xs_value: np.array,
    params: NeuralProcessParams,
    epsilon: Optional[tf.Tensor] = None,
    n_draws: int = 1,
) -> GaussianParams:
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

    # Inputs for prediction time
    xs = tf.constant(context_xs_value, dtype=tf.float32)
    ys = tf.constant(context_ys_value, dtype=tf.float32)
    x_star = tf.constant(input_xs_value, dtype=tf.float32)

    # For out-of-sample new points
    z_params = xy_to_z_params(xs, ys, params)

    # the source of randomness can be optionally passed as an argument
    if epsilon is None:
        epsilon = tf.random_normal((n_draws, params.dim_z))
    z_samples = tf.multiply(epsilon, z_params.sigma)
    z_samples = tf.add(z_samples, z_params.mu)

    y_star = decoder_g(z_samples, x_star, params)

    return y_star
