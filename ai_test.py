import tensorflow as tf
import numpy as np
from layers import calc_layers, conv2d, get_layer_data

x = np.random.random(5 * 5 * 3).reshape((5,5,3)).astype(np.float32)


model = tf.keras.Sequential([
    tf.keras.Input(shape=(5,5,3)),
    tf.keras.layers.Conv2D(2, 3, padding="same", activation="swish", bias_initializer='random_normal'),
    tf.keras.layers.Conv2D(2, 3, padding="same", activation="linear", bias_initializer='random_normal'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, bias_initializer='random_normal', activation="sigmoid")
])
model.compile()

y_model = model(np.array([x]))

layer_data = get_layer_data(model.layers)
y_np = calc_layers(x, layer_data).data1d[0]

print("y_model", y_model)
print("y_np", y_np)
print("diff", np.sum(np.abs(y_np - y_model)))