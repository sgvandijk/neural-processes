import random
from collections.__init__ import namedtuple
from typing import Dict

import numpy as np
import tensorflow as tf

NeuralProcessParams = namedtuple(
    "NeuralProcessParams",
    ["dim_x", "dim_y", "dim_r", "dim_z", "n_hidden_units_h", "n_hidden_units_g"],
)


def split_context_target(xs, ys, n_context):
    """Randomly split a set of x,y samples into context and target sets"""
    context_mask = np.zeros(xs.shape[0], dtype=bool)
    context_mask[[i for i in random.sample(range(xs.shape[0]), n_context)]] = True

    context_xs = xs[context_mask]
    context_ys = ys[context_mask]

    target_xs = xs[~context_mask]
    target_ys = ys[~context_mask]

    return context_xs, context_ys, target_xs, target_ys
