from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Dense
from tensorflow.keras.models import Sequential, Model
from neuralprocesses import NeuralProcessParams


class Encoder(tf.keras.Model):
    def __init__(self, params: NeuralProcessParams):
        super(Encoder, self).__init__(self)
        self.params = params
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1)

        self.ls = []
        for i, n_hidden_units in enumerate(self.params.n_hidden_units_h):
            l = Dense(
                n_hidden_units,
                activation="sigmoid",
                name=f"encoder_layer_{i}",
                kernel_initializer=initializer,
            )
            self.ls.append(l)

        i = len(self.params.n_hidden_units_h)
        l = Dense(
            self.params.dim_r,
            name=f"encoder_layer_{i}",
            kernel_initializer=initializer,
            bias_initializer=initializer,
        )
        self.ls.append(l)

        # Mu is simple linear
        self.z_mu = Dense(
            params.dim_z, name="z_params_mu", kernel_initializer=initializer
        )
        # Sigma should be possitive, use soft-plus
        self.z_sigma = Dense(
            params.dim_z,
            activation="softplus",
            name="z_params_sigma",
            kernel_initializer=initializer,
            bias_initializer=initializer,
        )

    def call(self, context_xs, context_ys):
        context_xys = tf.concat([context_xs, context_ys], axis=-1)
        n_c = tf.shape(context_xys)[-2]

        # Fold batch and samples together into superbatch, so that encoder
        # is the same per sample
        batch_size = tf.shape(context_xys)[0]
        context_xys = tf.reshape(
            context_xys, [batch_size * n_c, self.params.dim_x + self.params.dim_y]
        )
        context_xys.set_shape([None, self.params.dim_x + self.params.dim_y])

        x = context_xys
        for l in self.ls:
            x = l(x)

        # Unfold batch and samples
        rs = tf.reshape(x, [batch_size, n_c, self.params.dim_r])

        # Aggregate encodings
        aggregate_r = tf.reduce_mean(rs, axis=-2)

        # Map to latent distribution parameters
        mu = self.z_mu(aggregate_r)
        sigma = self.z_sigma(aggregate_r)

        return mu, sigma


def get_z_params(context_r, params: NeuralProcessParams) -> tf.Tensor:
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
    mu = Dense(params.dim_z, name="z_params_mu")(context_r)

    sigma = Dense(params.dim_z, name="z_params_sigma")(context_r)
    sigma = tf.nn.softplus(sigma)

    return [mu, sigma]


class Decoder(tf.keras.Model):
    def __init__(self, params: NeuralProcessParams):
        super(Decoder, self).__init__(self)
        self.params = params

        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1)

        self.ls = []
        # First layers are relu
        for i, n_hidden_units in enumerate(params.n_hidden_units_g):
            l = Dense(
                n_hidden_units,
                activation="sigmoid",
                name="decoder_layer_{}".format(i),
                kernel_initializer=initializer,
                bias_initializer=initializer,
            )
            self.ls.append(l)

        # Last layer is simple linear
        i = len(self.params.n_hidden_units_g)
        l = Dense(
            self.params.dim_y,
            name="decoder_layer_{}".format(i),
            kernel_initializer=initializer,
            bias_initializer=initializer,
        )
        self.ls.append(l)

    def call(self, x_star, z_samples):
        batch_size = tf.shape(x_star)[0]

        n_x = tf.shape(x_star)[-2]  # bs X nx X dim_x
        n_z = tf.shape(z_samples)[-2]  # bs X nz X dim_z

        x_star = tf.expand_dims(x_star, 1)
        x_star = tf.repeat(x_star, n_z, axis=1)

        z_samples = tf.expand_dims(z_samples, 2)
        z_samples = tf.repeat(z_samples, n_x, axis=2)

        inputs = tf.concat([x_star, z_samples], axis=-1)

        # Fold batch and samples together into superbatch, so that decoder
        # is the same per sample
        inputs = tf.reshape(
            inputs, [batch_size * n_z * n_x, self.params.dim_x + self.params.dim_z]
        )
        inputs.set_shape([None, self.params.dim_x + self.params.dim_z])

        x = inputs
        for l in self.ls:
            x = l(x)

        # Unfold batch and samples
        mu_star = tf.reshape(x, [batch_size, n_z, n_x, self.params.dim_y])

        # TODO: TF 2 doesn't like adding a constant to output
        # sigma_star = tf.constant(noise_std, dtype=tf.float32)

        return mu_star


def xy_to_z_params(context_xs, context_ys, params: NeuralProcessParams,) -> Model:
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
    xys = tf.concat([context_xs, context_ys], axis=-1)
    rs = encoder_h(xys, params)
    r = aggregate_r(rs, params)
    z = get_z_params(r, params)
    return z
