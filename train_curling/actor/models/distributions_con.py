from typing import MutableMapping
import tensorflow as tf

import numpy as np


class CategoricalPd:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        
    def mode(self):
        return tf.argmax(self.logits, axis=-1)

    def logp(self, x):
        return -self.neglogp(x)

    # def neglogp(self, x):
    #     # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
    #     # Note: we can't use sparse_softmax_cross_entropy_with_logits because
    #     #       the implementation does not allow second-order derivatives...
    #     if x.dtype in {tf.uint8, tf.int32, tf.int64}:
    #         # one-hot encoding
    #         x_shape_list = x.shape.as_list()
    #         logits_shape_list = self.mu.get_shape().as_list()[:-1]
    #         for xs, ls in zip(x_shape_list, logits_shape_list):
    #             if xs is not None and ls is not None:
    #                 assert xs == ls, 'shape mismatch: {} in x vs {} in mu'.format(xs, ls)

    #         x = tf.one_hot(x, self.mu.get_shape().as_list()[-1])
    #     else:
    #         # already encoded
    #         assert x.shape.as_list() == self.mu.shape.as_list()

    #     return tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.mu, labels=x)

    def kl(self, other):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

    def sample(self):
        # pi = tf.compat.v1.distributions.Normal(loc=self.mu, scale=tf.nn.softplus(self.sigma))
        pi = tf.compat.v1.distributions.Normal(loc=self.mu, scale=0.1)
        a = tf.squeeze(pi.sample(1), axis=0)
        a = tf.compat.v1.clip_by_value(a, -1, 1)
        return a

    def neglogp(self, x):
        # pi = tf.compat.v1.distributions.Normal(loc=self.mu, scale=tf.nn.softplus(self.sigma))
        pi = tf.compat.v1.distributions.Normal(loc=self.mu, scale=0.1)
        ratio = pi.prob(x)+1e-5
        neglogratio = tf.reduce_sum(-tf.log(ratio),-1)
        return neglogratio
