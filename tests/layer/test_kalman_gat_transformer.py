import numpy as np
from tensorflow.keras import Model

from stellargraph.layer.kalman_gat_transformer import KalmanGATTransformer


def get_timeseries_graph_data():
    featuresX = np.random.rand(4, 5, 3)
    featuresY = np.random.rand(4, 5)
    adj = np.random.randint(0, 2, size=(5, 5))
    return featuresX, featuresY, adj


def test_kalman_gat_transformer_model():
    fx, fy, a = get_timeseries_graph_data()
    model_layer = KalmanGATTransformer(seq_len=fx.shape[-1], adj=a, gat_layer_sizes=[2], transformer_layers=1)
    x_input, x_output = model_layer.in_out_tensors()
    model = Model(inputs=x_input, outputs=x_output)
    model.compile(optimizer="adam", loss="mae")
    history = model.fit(fx, fy, epochs=2, batch_size=1, verbose=0)
    assert len(history.history["loss"]) == 2

