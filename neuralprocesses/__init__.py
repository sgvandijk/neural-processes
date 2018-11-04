import random
from collections.__init__ import namedtuple
from typing import Dict

import numpy as np
import tensorflow as tf

GaussianParams = namedtuple('GaussianParams', ['mu', 'sigma'])


def split_context_target(xs: np.array, ys: np.array, n_context: int,
                         context_xs: tf.Tensor, context_ys: tf.Tensor, target_xs: tf.Tensor, target_ys: tf.Tensor) \
        -> Dict[tf.Tensor, np.array]:
    """Split samples randomly into context and target sets
    """
    indices = set(range(len(ys)))
    context_set_indices = set(random.sample(indices, n_context))
    target_set_indices = indices - context_set_indices

    return {
        context_xs: xs[list(context_set_indices)].reshape((-1, 1)),
        context_ys: ys[list(context_set_indices)].reshape((-1, 1)),
        target_xs: xs[list(target_set_indices)].reshape((-1, 1)),
        target_ys: ys[list(target_set_indices)].reshape((-1, 1))
    }
