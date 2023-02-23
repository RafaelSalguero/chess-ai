import tensorflow as tf
import numpy as np
from eval import evalBoard

from train import get_sim_games

import os

from utils import onehot_encode_board
from view import print_board
# Disable GPU training:
tf.config.set_visible_devices([], 'GPU')

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
(x_train, y_train, ply_train) = get_train_data(0, 0, 3000, 50000, False)

print("generating test data")

(x_test, y_test, ply_test) = get_train_data(0, 0, 3000, 10000, False)

print(f"train sample {x_train.shape[0]} {y_train.shape[0]}:")

for i in np.random.random_integers(0, x_train.shape[0] - 1, 10000):
    if(evalBoard(x_train[i], 1) != y_train[i]):
        print(f"EVAL DIFF y_train: {y_train[i]} evalBoard: {evalBoard(x_train[i], 1)}")
        exit()
    # print(f"ply: {ply_train[i]}, eval: {y_train[i]} evalBoard: {evalBoard(x_train[i], 1)}")


print("test sample:")

for i in np.random.random_integers(0, x_test.shape[0], 20):
    print_board(x_test[i])
    print(f"eval: {y_test[i]}")

x_test = onehot_encode_board(x_test)
x_train = onehot_encode_board(x_train)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(8,8,8)),
    tf.keras.layers.Conv2D(8, 1, activation="swish"),
    tf.keras.layers.Conv2D(8, 1, activation="swish"),
    tf.keras.layers.Conv2D(8, 1, activation="swish"),
    
    tf.keras.layers.Conv2D(1, 1),
    tf.keras.layers.AveragePooling2D(pool_size=(8,8)),
    
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(1)
])

print(model.summary())

model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])

print("train size: " + str(x_train.shape[0]))

model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=1000, batch_size=32)


model.evaluate(x_train,  y_train, verbose=2)
model.evaluate(x_test,  y_test, verbose=2)

exit()
y_eval = model(x_test)
print("eval sample:")
for i in np.random.random_integers(0, x_test.shape[0] - 1, 100):
    print_board(x_test[i])
    print(f"test: {y_test[i]} evalBoard: {evalBoard(x_test[i], 1)} value: {y_eval[i]}")



model.save("models/alpha_beta_3")