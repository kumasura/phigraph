# -*- coding: utf-8 -*-
"""Kalman GAT Transformer layer."""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dropout, Dense, MultiHeadAttention

from .gat_transformer import GATTransformer


class KalmanFilterAttention(Layer):
    """A simple learnable Kalman filter with attention."""

    def __init__(self, units, num_heads=1, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.dropout = Dropout(dropout)
        self.state_dense = Dense(units)
        self.obs_dense = Dense(units)
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=units, dropout=dropout)
        self.out_dense = Dense(units)

    def call(self, inputs):
        # inputs: B x T x F
        inputs_T = tf.transpose(inputs, [1, 0, 2])  # T x B x F
        batch_size = tf.shape(inputs)[0]
        init_state = tf.zeros((batch_size, self.units))

        def step(prev, obs):
            pred = self.state_dense(prev)
            obs_p = self.obs_dense(obs)
            attn = self.attn(tf.expand_dims(pred, 1), tf.expand_dims(obs_p, 1), tf.expand_dims(obs_p, 1))
            attn = tf.squeeze(attn, 1)
            new_state = pred + attn
            return new_state

        states = tf.scan(step, inputs_T, initializer=init_state)
        states = tf.transpose(states, [1, 0, 2])  # B x T x units
        states = self.dropout(states)
        return self.out_dense(states)


class KalmanGATTransformer(GATTransformer):
    """GAT Transformer preceded by a learnable Kalman filter."""

    def __init__(self, *args, kalman_units=None, kalman_heads=1, kalman_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if kalman_units is None:
            kalman_units = self.variates
        self.kalman_filter = KalmanFilterAttention(kalman_units, num_heads=kalman_heads, dropout=kalman_dropout)

    def _apply_kalman(self, x):
        # x: B x N x T x V
        def fn(t):
            t = tf.reshape(t, (-1, self.seq_len, self.variates))
            t = self.kalman_filter(t)
            return tf.reshape(t, (-1, self.n_nodes, self.seq_len, self.variates))

        return tf.keras.layers.Lambda(fn)(x)

    def __call__(self, x):
        x_in, out_idx = x
        h = x_in
        if not self.multivariate_input:
            h = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(h)
        h = self._apply_kalman(h)
        if not self.multivariate_input:
            h = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(h)
        return super().__call__((h, out_idx))
