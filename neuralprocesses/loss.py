import tensorflow as tf

from neuralprocesses import GaussianParams


def KLqp_gaussian(mu_q: tf.Tensor, sigma_q: tf.Tensor, mu_p: tf.Tensor, sigma_p: tf.Tensor) -> tf.Tensor:
    """Kullback-Leibler divergence between two Gaussian distributions

    Determines KL(q || p) = < log( q / p ) >_q

    Parameters
    ----------
    mu_q
        Mean tensor of distribution q, shape: (1, dim)
    sigma_q
        Variance tensor of distribution q, shape: (1, dim)
    mu_p
        Mean tensor of distribution p, shape: (1, dim)
    sigma_p
        Variance tensor of distribution p, shape: (1, dim)

    Returns
    -------
        KL tensor, shape: (1)
    """
    sigma2_q = tf.square(sigma_q) + 1e-16
    sigma2_p = tf.square(sigma_p) + 1e-16
    temp = sigma2_q / sigma2_p + tf.square(mu_q - mu_p) / sigma2_p - 1.0 + tf.log(sigma2_p / sigma2_q + 1e-16)
    return 0.5 * tf.reduce_sum(temp)


def loglikelihood(y_star: tf.Tensor, y_pred_params: GaussianParams):
    """Log-likelihood of an output given a predicted """
    p_normal = tf.distributions.Normal(loc=y_pred_params.mu, scale=y_pred_params.sigma)
    loglike = p_normal.log_prob(y_star)
    loglike = tf.reduce_sum(loglike, axis=0)
    loglike = tf.reduce_mean(loglike)
    return loglike