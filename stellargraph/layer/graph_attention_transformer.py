# -*- coding: utf-8 -*-
"""Graph Attention Transformer Layer."""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from .graph_attention import GraphAttention, GraphAttentionSparse
from .misc import SqueezedSparseConversion

__all__ = ["GraphAttentionTransformer"]


class GraphAttentionTransformer(layers.Layer):
    """Stack Graph Attention with a Transformer block.

    This layer first applies a :class:`.GraphAttention` layer to the input
    features and then processes the resulting representations using a
    Transformer-style multi-head self-attention layer. The outputs of the GAT
    and Transformer blocks are combined using an aggregation function.

    The ``aggregator`` argument can be a string identifying one of the builtin
    aggregation methods (``"add"``, ``"mean"`` or ``"concat"``) or a custom
    callable that combines two tensors ``(gat_out, transformer_out)``. When a
    callable is supplied, :meth:`compute_output_shape` infers the final feature
    dimension by applying the aggregator to dummy tensors of the GAT output
    shape.
    """

    def __init__(
        self,
        units,
        attn_heads=1,
        attn_heads_reduction="concat",
        aggregator="add",
        transformer_heads=1,
        in_dropout_rate=0.0,
        attn_dropout_rate=0.0,
        activation="relu",
        use_sparse=False,
        **kwargs,
    ):
        self.units = units
        self.attn_heads = attn_heads
        self.attn_heads_reduction = attn_heads_reduction
        self.aggregator = aggregator
        self.transformer_heads = transformer_heads
        self.in_dropout_rate = in_dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.activation = activation
        self.use_sparse = use_sparse

        gat_class = GraphAttentionSparse if use_sparse else GraphAttention
        self.gat = gat_class(
            units=units,
            attn_heads=attn_heads,
            attn_heads_reduction=attn_heads_reduction,
            in_dropout_rate=in_dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            activation=activation,
        )

        self._set_agg_fn(aggregator)

        self.transformer = layers.MultiHeadAttention(
            num_heads=transformer_heads,
            key_dim=self.gat.output_dim,
            dropout=in_dropout_rate,
        )
        self.dropout = layers.Dropout(in_dropout_rate)
        super().__init__(**kwargs)

    def build(self, input_shapes):
        # delegate building to the internal GAT layer
        self.gat.build(input_shapes)
        self.built = True

    def compute_output_shape(self, input_shapes):
        gat_shape = self.gat.compute_output_shape(input_shapes)
        out_dim = gat_shape[-1]

        if isinstance(self.aggregator, str):
            if self.aggregator == "concat":
                out_dim *= 2
        else:
            # determine the final dimension by applying the aggregator
            dummy = tf.zeros((1, 1, out_dim))
            out_dim = self._agg_fn(dummy, dummy).shape[-1]

        return (gat_shape[0], gat_shape[1], out_dim)

    def _set_agg_fn(self, agg):
        if isinstance(agg, str):
            if agg == "add":
                self._agg_fn = lambda a, b: a + b
            elif agg == "mean":
                self._agg_fn = lambda a, b: (a + b) / 2.0
            elif agg == "concat":
                self._agg_fn = lambda a, b: K.concatenate([a, b], axis=-1)
            else:
                raise ValueError(f"Unknown aggregator '{agg}'")
        elif callable(agg):
            self._agg_fn = agg
        else:
            raise TypeError("aggregator must be a string or callable")

    def get_config(self):
        return {
            "units": self.units,
            "attn_heads": self.attn_heads,
            "attn_heads_reduction": self.attn_heads_reduction,
            "aggregator": self.aggregator,
            "transformer_heads": self.transformer_heads,
            "in_dropout_rate": self.in_dropout_rate,
            "attn_dropout_rate": self.attn_dropout_rate,
            "activation": self.activation,
            "use_sparse": self.use_sparse,
        }

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            raise TypeError(f"inputs: expected list, found {type(inputs).__name__}")

        x_in, *A = inputs

        if self.use_sparse:
            if len(A) != 2:
                raise ValueError(
                    "Sparse mode requires indices and values for adjacency"
                )
            A_indices, A_values = A
            A_tensor = SqueezedSparseConversion(shape=(x_in.shape[1], x_in.shape[1]))(
                [A_indices, A_values]
            )
            gat_out = self.gat([x_in, A_tensor])
        else:
            if len(A) != 1:
                raise ValueError("Dense mode requires a single adjacency matrix")
            gat_out = self.gat([x_in, A[0]])

        trans_out = self.transformer(gat_out, gat_out)
        trans_out = self.dropout(trans_out)

        return self._agg_fn(gat_out, trans_out)
