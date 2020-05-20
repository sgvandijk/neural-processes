import tensorflow as tf
import tensorflow_probability as tfp


def loglikelihood(target_ys, pred_ys_mu, pred_scale=0.25):
    """Log-likelihood of predictions given target

    Determines log( p(target_y | z, target_x) ),
    from the predictions that were decoded given samples of z.

    Parameters
    ----------
    target_ys
        Tensor with shape: (bs, n_targets, dim_y) - single output vector per target sample
    pred_ys
        Tensor with shape: (bs, n_samples, n_targets, dim_y ) - several samples per target sample
    pred_scale
        The fixed output variance
    """

    p_normal = tfp.distributions.Normal(loc=pred_ys_mu, scale=pred_scale)
    loglike = p_normal.log_prob(target_ys)
    # Sum over targets
    loglike = tf.reduce_sum(loglike, axis=-2)
    # Mean over samples of z and batch
    loglike = tf.reduce_mean(loglike)
    return loglike


def kullback_leibler_gaussian(mu_q, sigma_q, mu_p, sigma_p):
    """Kullback-Leibler divergence between two Gaussian distributions

    Determines KL(q || p) = < log( q / p ) >_q

    See:
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Examples

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
    temp = (
        sigma2_q / sigma2_p
        + tf.math.square(mu_q - mu_p) / sigma2_p
        - 1.0
        + tf.math.log(sigma2_p / sigma2_q + 1e-16)
    )
    return 0.5 * tf.reduce_sum(temp)
