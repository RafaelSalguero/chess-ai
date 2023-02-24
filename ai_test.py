import tensorflow as tf
import numpy as np
from layers import calc_layers, conv2d, get_layer_data

x = np.random.random(5 * 5 * 3).reshape((5,5,3)).astype(np.float32)


model = tf.keras.Sequential([
    tf.keras.Input(shape=(5,5,3)),
    tf.keras.layers.Conv2D(2, 3, padding="same", activation="linear"),
    tf.keras.layers.Conv2D(2, 3, padding="same", activation="linear"),
])
model.compile()

y_model = model(np.array([x]))

layer_data = get_layer_data(model.layers)
y_np = calc_layers(x, layer_data)

print("y_model", y_model)
print("y_np", y_np)
print("diff", np.sum(np.abs(y_np - y_model)))