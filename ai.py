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

print("generating test data")

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
(x_train, y_train) = get_train_data(2, 50000, 10000, False)


(x_test, y_test) = get_train_data(2, 50000, 1000)

# np.savez_compressed("train_data/get_sim_games_2_16384", x_train = x_train, y_train=y_train)
x_train = onehot_encode_board(x_train)
x_test = onehot_encode_board(x_test)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(8,8,8)),
    tf.keras.layers.Conv2D(1,1, padding="same", activation="linear"),
    
    tf.keras.layers.AveragePooling2D(pool_size=(8,8)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1)
])

print(model.summary())

loss_fn = tf.keras.losses.MeanSquaredError()
model.compile(optimizer='adam', loss=loss_fn, metrics=[tf.keras.metrics.RootMeanSquaredError()])

print("train size: " + str(x_train.shape[0]))

model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=1000, batch_size=32, callbacks=[
        tf.keras.callbacks.EarlyStopping("loss", min_delta=5, patience=20, mode="min")
])

model.evaluate(x_train,  y_train, verbose=2)
model.evaluate(x_test,  y_test, verbose=2)
