import tensorflow as tf
import numpy as np

from train import get_sim_games

import os

from utils import onehot_encode_board
from view import print_board
# Disable GPU training:
# tf.config.set_visible_devices([], 'GPU')

# Chess board representation will be a 8x8x6 where the 3rd dimension, each one represents a piece type
# Piece types: 0 Pawn, 1 Knight,  2 Bishop, 3 Rook, 4 Queen, 5 King


def get_train_data(depth, max_iter, size, cache = False):
    file_name = f'train_data/get_sim_games_{depth}_{max_iter}_{size}.npz'
    if(cache and os.path.isfile(file_name)):
        with np.load(file_name) as data:
            return (data['x'], data['y'])
        
    (x, y) = get_sim_games(depth, max_iter, size=size, threads=8, verbose= False)

    if (cache):
        np.savez_compressed(file_name, x=x, y=y)
    return (x, y)

# data = np.load("train_data/get_sim_games_2_16384.npz")
print("generating train data")
(x_train, y_train) = get_train_data(1, 50000, 50000, False)

print("generating test data")

(x_test, y_test) = get_train_data(1, 50000, 10000, False)

print("train sample:")

for i in np.random.random_integers(0, x_train.shape[0], 20):
    print_board(x_train[i])
    print(f"eval: {y_train[i]}")


print("test sample:")

for i in np.random.random_integers(0, x_test.shape[0], 20):
    print_board(x_test[i])
    print(f"eval: {y_test[i]}")

print("one-hot encoding")

# x_train = onehot_encode_board(x_train)
# x_test = onehot_encode_board(x_test)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(8,8,1)),
    tf.keras.layers.Conv2D(8, 15, padding="same", activation="swish"),
    tf.keras.layers.Conv2D(8, 15, padding="same", activation="swish"),
    tf.keras.layers.Conv2D(8, 15, padding="same", activation="swish"),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation="swish"),

    tf.keras.layers.Dense(1)
])

print(model.summary())

loss_fn = tf.keras.losses.MeanSquaredError()
model.compile(optimizer='adam', loss=loss_fn, metrics=[tf.keras.metrics.RootMeanSquaredError()])

print("train size: " + str(x_train.shape[0]))

model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=1000, batch_size=256, callbacks=[
        tf.keras.callbacks.EarlyStopping("loss", min_delta=5, patience=20, mode="min")
])

model.evaluate(x_train,  y_train, verbose=2)
model.evaluate(x_test,  y_test, verbose=2)
