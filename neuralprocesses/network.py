from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Dense
from tensorflow.keras.models import Sequential, Model
from neuralprocesses import GaussianParams, NeuralProcessParams


def encoder_h(params: NeuralProcessParams) -> Model:
    """Map context inputs (x_i, y_i) to r_i

    Creates a fully connected network with a single sigmoid hidden
    layer and linear output layer.

    Parameters
    ----------
    params
        Neural process parameters

    Returns
    -------
        Output tensor of encoder network

    """
    context_xys = Input(params.dim_x + params.dim_y)
    x = context_xys

    # First layers are relu
    for i, n_hidden_units in enumerate(params.n_hidden_units_h):
        l = Dense(n_hidden_units, activation=tf.nn.relu, name=f"encoder_layer_{i}")
        x = l(x)

    # Last layer is simple linear
    i = len(params.n_hidden_units_h)
    r = Dense(params.dim_r, name=f"encoder_layer_{i}")
    Model(context_xys, r)


def aggregate_r(params: NeuralProcessParams) -> Model:
    """Aggregate the output of the encoder to a single representation

    Creates an aggregation (mean) operator to combine the encodings of
    multiple context inputs

    Parameters
    ----------
    context_inputs
        Input encodings tensor, shape: (n_samples, dim_r)

    Returns
    -------
        Output tensor of aggregation result

    """
    context_xys = Input([None, params.dim_r])
    mean = tf.reduce_mean(context_rs, axis=0)
    output = tf.reshape(mean, [1, -1])
    return Model(context_rs, output)


def get_z_params(params: NeuralProcessParams) -> Model:
    """Map encoding to mean and covariance of the random variable Z

    Creates a linear dense layer to map encoding to mu_z, and another
    linear mapping + a softplus activation for Sigma_z

    Parameters
    ----------
    context_r
        Input encoding tensor, shape: (1, dim_r)
    params
        Neural process parameters

    Returns
    -------
        Output tensors of the mappings for mu_z and Sigma_z

    """
    context_r = Input(params.dim_r)
    mu = Dense(params.dim_z, name="z_params_mu")(context_r)

    sigma = Dense(params.dim_z, name="z_params_sigma")(context_r)
    sigma = tf.nn.softplus(sigma)

    return Model(context_r, [mu, sigma])


def decoder_g(params: NeuralProcessParams, noise_std: float = 0.05,) -> GaussianParams:
    """Determine output y* by decoding input and latent variable

    Creates a fully connected network with a single sigmoid hidden
    layer and linear output layer.

    Parameters
    ----------
    z_samples
        Random samples from the latent variable distribution, shape: (n_z_draws, dim_z)
    input_xs
        Input values to predict for, shape: (n_x_samples, dim_x)
    params
        Neural process parameters
    noise_std
        Constant standard deviation used on output

    Returns
    -------
        Output tensors for the parameters of Gaussian distributions for target outputy, where its mean mu has shape
        (n_x_samples, n_z_draws)
        TODO: this assumes/forces dim_y = 1

    """
    # N samples of size Dx
    x_star = Input([None, params.dim_x])
    n_x = tf.shape(x_star)[-2]

    # Single latent sample of size Dz
    z_sample = Input(params.dim_z)
    z_sample_ = tf.expand_dims(z_sample, -2)
    z_samples = tf.repeat(z_sample_, n_x, axis=-2)
    inputs = tf.concat([x_star, z_samples], axis=-1)

    hidden_layer = inputs
    # First layers are relu
    for i, n_hidden_units in enumerate(params.n_hidden_units_g):
        hidden_layer = Dense(
            n_hidden_units,
            activation="sigmoid",
            name="decoder_layer_{}".format(i),
            kernel_initializer="normal",
        )(hidden_layer)

    # Last layer is simple linear
    i = len(params.n_hidden_units_g)
    mu_star = Dense(1, name="decoder_layer_{}".format(i),)(hidden_layer)

    # TODO: TF 2 doesn't like adding a constant to output
    # sigma_star = tf.constant(noise_std, dtype=tf.float32)

    return Model([x_star, z_sample], mu_star)


@tf.function
def xy_to_z_params(params: NeuralProcessParams) -> Model:
    """Wrapper to create full graph from context samples to parameters of pdf of Z

    Parameters
    ----------
    context_xs
        Tensor with context features, shape: (n_samples, dim_x)
    context_ys
        Tensor with context targets, shape: (n_samples, dim_y)
    params
        Neural process parameters

    Returns
    -------
        Output tensors of the mappings for mu_z and Sigma_z
    """
    context_xs = Input(params.dim_x)
    context_ys = Input(params.dim_y)

    xys = tf.concat([context_xs, context_ys], axis=1)
    rs = encoder_h(params)(xys)
    r = aggregate_r(params)(rs)
    z_params = get_z_params(params)(r)
    return Model([context_xs, context_ts], z_params)
