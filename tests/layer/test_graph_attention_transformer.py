import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input

from stellargraph.layer import GraphAttentionTransformer


class TestGraphAttentionTransformer:
    N = 4
    F_in = 3

    def get_inputs(self):
        x = Input(batch_shape=(1, self.N, self.F_in))
        A = Input(batch_shape=(1, self.N, self.N))
        return [x, A]

    def test_output_shapes_concat(self):
        layer = GraphAttentionTransformer(
            units=2,
            attn_heads=1,
            attn_heads_reduction="average",
            aggregator="concat",
            transformer_heads=1,
        )
        x_inp = self.get_inputs()
        out = layer(x_inp)
        assert out.shape.as_list() == [1, self.N, 4]

    def test_custom_aggregator(self):
        agg = lambda inputs0, inputs1: inputs0 - inputs1
        layer = GraphAttentionTransformer(
            units=2,
            attn_heads=1,
            attn_heads_reduction="average",
            aggregator=agg,
            transformer_heads=1,
        )
        x_inp = self.get_inputs()
        out = layer(x_inp)
        assert out.shape.as_list() == [1, self.N, 2]

    def test_custom_aggregator_dimension_change(self):
        def agg(a, b):
            c = tf.concat([a, b], axis=-1)
            return c[:, :, :-1]

        layer = GraphAttentionTransformer(
            units=2,
            attn_heads=1,
            attn_heads_reduction="average",
            aggregator=agg,
            transformer_heads=1,
        )

        x_inp = self.get_inputs()
        out = layer(x_inp)
        assert out.shape.as_list() == [1, self.N, 3]
        assert layer.compute_output_shape([t.shape for t in x_inp]) == (1, self.N, 3)
