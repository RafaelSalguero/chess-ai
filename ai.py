from minmax import alphabeta, iterative_deepening, inf_val
import tensorflow as tf

import numpy as np
from eval import evalBoard

from train import get_sim_games

import os

from utils import cap_histogram, onehot_encode_board
from view import print_board
# Disable GPU training:
# tf.config.set_visible_devices([], 'GPU')

# Chess board representation will be a 8x8x6 where the 3rd dimension, each one represents a piece type
# Piece types: 0 Pawn, 1 Knight,  2 Bishop, 3 Rook, 4 Queen, 5 King


def get_train_data(depth, quiescence_depth, max_iter, size, cache = False):
    file_name = f'train_data/get_sim_games_{depth}_{max_iter}_{size}.npz'
    if(cache and os.path.isfile(file_name)):
        with np.load(file_name) as data:
            return (data['x'], data['y'], data['ply'])
        
    (x, y, ply) = get_sim_games(depth, quiescence_depth, max_iter, size=size, threads=8, verbose=False)

    if (cache):
        np.savez_compressed(file_name, x=x, y=y, ply=ply)
    return (x, y, ply)

# data = np.load("train_data/get_sim_games_2_16384.npz")
print("generating train data")
(x_train, y_train, ply_train) = get_train_data(10, 5, 100, 10000000, True)

indices = cap_histogram(y_train, 2000, 1)

print(f"original len: {len(y_train)}")
x_train = x_train[indices]
y_train = y_train[indices]

print(f"capped len: {len(y_train)}")

print("generating test data")

(x_test, y_test, ply_test) = get_train_data(10, 5, 100, 10000, True)

print(f"train sample {x_train.shape[0]} {y_train.shape[0]}:")


def signed_log(x):
    return tf.math.log(tf.math.abs(x) + 1) * tf.math.sign(x)

def inv_signed_log(x):
    return (tf.math.exp(tf.math.abs(x)) - 1) * tf.math.sign(x)

print("train sample:")
for i in np.random.random_integers(0, x_train.shape[0], 20):
    print_board(x_train[i])
    print(f"eval: {y_train[i]}")

print("test sample:")

for i in np.random.random_integers(0, x_test.shape[0], 20):
    print_board(x_test[i])
    print(f"eval: {y_test[i]}")

model_x_test = onehot_encode_board(x_test)
model_x_train = onehot_encode_board(x_train)
model_y_train = y_train
model_y_test = y_test

def create_model():
    inputs = tf.keras.Input(shape=(8,8,8))

    x = inputs

    x = tf.keras.layers.Conv2D(16, 7, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(16, 7, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(16, 7, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(16, 7, padding="same", activation="relu")(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)

    x = tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu")(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(8, activation = "relu")(x)

    x = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs = inputs, outputs = x)
    return model


model = create_model()


print(model.summary())

model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])

print("train size: " + str(x_train.shape[0]))

model.fit(model_x_train, model_y_train, validation_data = (model_x_test, model_y_test), epochs=200, batch_size=128)

model.save("models/alpha_beta_3")

model.evaluate(model_x_train,  model_y_train, verbose=2)
model.evaluate(model_x_test,  model_y_test, verbose=2)

y_eval = model(model_x_test)
print("eval sample:")
for i in np.random.random_integers(0, x_test.shape[0] - 1, 100):
    print_board(x_test[i])
    print(f"test: {y_test[i]} evalBoard: {evalBoard(x_test[i], 1)} value: {y_eval[i]}")

