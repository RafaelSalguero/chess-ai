import tensorflow as tf
import numpy as np
from layers import conv2d

x = np.random.random(5 * 5 * 3).reshape((5,5,3))


model = tf.keras.Sequential([
    tf.keras.Input(shape=(5,5,3)),
    tf.keras.layers.Conv2D(2, 3, padding="same")
])
model.compile()

y_model = model(np.array([x]))

w = model.layers[0].weights
kernel = w[0].numpy()
bias = w[1].numpy()

print("kernel: ", kernel)
print("bias: ", bias)

y_np = conv2d(x, np.zeros((5,5,2)), kernel, bias)

print("y_model", y_model)
print("y_np", y_np)
print("diff", np.sum(np.abs(y_np - y_model)))