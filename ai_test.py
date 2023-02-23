import tensorflow as tf
import numpy as np

def y_func(x):
    real_val = 0.7 if x == 0 else 0.2 if x == 1 else 0.5
    return real_val
    return 1 if np.random.random() < real_val else 0

x_train = np.random.random_integers(0, 2, 1000)
y_train = np.array(list(map(y_func, x_train)))

x_test = np.array([0, 1, 2])
y_test = np.array([0.7, 0.2, 0.5])

print(x_test)
print(y_test)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1)),
    tf.keras.layers.Dense(3, activation="swish"),
    tf.keras.layers.Dense(3, activation="swish"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

print(model.summary())

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=100, batch_size=32)

print("actual output:")
print(model(np.array([[0], [1], [2]])))