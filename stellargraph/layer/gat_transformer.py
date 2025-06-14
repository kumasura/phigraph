# -*- coding: utf-8 -*-
"""GAT + Transformer layer for time series forecasting."""

import tensorflow as tf
from tensorflow.keras import activations, initializers, constraints, regularizers
from tensorflow.keras.layers import Layer, Dropout, Dense, MultiHeadAttention, LayerNormalization, Input, Permute, Reshape


class FixedAdjacencyGraphAttention(Layer):
    """A simplified Graph Attention layer with a fixed adjacency matrix."""

    def __init__(
        self,
        units,
        A,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.A = tf.cast(A, tf.float32)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        f = int(input_shape[-1])
        self.kernel = self.add_weight(
            shape=(f, self.units), initializer=self.kernel_initializer, name="kernel"
        )
        self.attn_self = self.add_weight(shape=(self.units, 1), initializer="glorot_uniform", name="attn_self")
        self.attn_neigh = self.add_weight(shape=(self.units, 1), initializer="glorot_uniform", name="attn_neigh")
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,), initializer=self.bias_initializer, name="bias")
        else:
            self.bias = None
        super().build(input_shape)

    def call(self, features):
        # features: B x N x F
        h = tf.tensordot(features, self.kernel, axes=1)  # B x N x units
        attn_for_self = tf.tensordot(h, self.attn_self, axes=1)  # B x N x 1
        attn_for_neighs = tf.tensordot(h, self.attn_neigh, axes=1)  # B x N x 1
        e = attn_for_self + tf.transpose(attn_for_neighs, [0, 2, 1])  # B x N x N
        e = tf.nn.leaky_relu(e)
        mask = tf.where(self.A > 0, tf.zeros_like(self.A), tf.ones_like(self.A) * -1e9)
        e = e + mask  # broadcast over batch
        attn = tf.nn.softmax(e, axis=-1)
        output = tf.matmul(attn, h)  # B x N x units
        if self.use_bias:
            output = output + self.bias
        return self.activation(output)


class TransformerBlock(Layer):
    """A basic Transformer encoder block."""

    def __init__(self, units, num_heads=2, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=units, dropout=dropout)
        self.ffn = tf.keras.Sequential([
            Dense(units, activation="relu"),
            Dense(units),
        ])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(dropout)

    def call(self, inputs):
        x = self.att(inputs, inputs)
        x = self.dropout(x)
        x = self.norm1(inputs + x)
        y = self.ffn(x)
        y = self.dropout(y)
        return self.norm2(x + y)


class GATTransformer:
    """Model combining Graph Attention layers with Transformer blocks."""

    def __init__(
        self,
        seq_len,
        adj,
        gat_layer_sizes,
        transformer_layers=1,
        gat_activations=None,
        aggregator="add",
        generator=None,
        dropout=0.5,
    ):
        if generator is not None:
            adj = generator.graph.to_adjacency_matrix(weighted=True).todense()
            seq_len = generator.window_size
            variates = generator.variates
        else:
            variates = None
        self.adj = adj
        self.n_nodes = adj.shape[0]
        self.seq_len = seq_len
        self.variates = variates if variates is not None else 1
        self.multivariate_input = variates is not None
        self.outputs = self.n_nodes * self.variates
        if gat_activations is None:
            gat_activations = ["relu"] * len(gat_layer_sizes)
        self._gat_layers = [
            FixedAdjacencyGraphAttention(
                units=self.seq_len * self.variates,
                A=self.adj,
                activation=act,
            )
            for act in gat_activations
        ]
        self._transformer_layers = [
            TransformerBlock(self.n_nodes * self.variates, dropout=dropout)
            for _ in range(transformer_layers)
        ]
        self.dropout = dropout
        self._decoder_layer = Dense(self.outputs, activation="sigmoid")
        self._set_agg_fn(aggregator)

    def _set_agg_fn(self, agg):
        if isinstance(agg, str):
            if agg == "add":
                self._agg_fn = lambda a, b: a + b
            elif agg == "mean":
                self._agg_fn = lambda a, b: (a + b) / 2.0
            elif agg == "concat":
                self._agg_fn = lambda a, b: tf.concat([a, b], axis=-1)
            else:
                raise ValueError(f"Unknown aggregator '{agg}'")
        elif callable(agg):
            self._agg_fn = agg
        else:
            raise TypeError("aggregator must be a string or callable")

    def __call__(self, x):
        x_in, _ = x
        h = x_in
        if not self.multivariate_input:
            h = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(h)
        h = Reshape((self.n_nodes, self.seq_len * self.variates))(h)
        for layer in self._gat_layers:
            h = layer(h)
        h = Reshape((self.n_nodes, self.seq_len, self.variates))(h)
        h = Permute((2, 1, 3))(h)
        h = Reshape((self.seq_len, self.n_nodes * self.variates))(h)
        h_in = h
        for layer in self._transformer_layers:
            h = layer(h)
        h = self._agg_fn(h_in, h)
        h = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=0, keepdims=True))(h)
        h = Dropout(self.dropout)(h)
        h = self._decoder_layer(h)
        if self.multivariate_input:
            h = Reshape((self.n_nodes, self.variates))(h)
        return h

    def in_out_tensors(self):
        if self.multivariate_input:
            shape = (None, self.n_nodes, self.seq_len, self.variates)
        else:
            shape = (None, self.n_nodes, self.seq_len)
        x_t = Input(batch_shape=shape)
        out_indices_t = Input(batch_shape=(None, self.n_nodes), dtype="int32")
        x_out = self([x_t, out_indices_t])
        return x_t, x_out
