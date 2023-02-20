import tensorflow as tf
import numpy as np
from ai_arch import arch_a0, arch_a1_c0, arch_a1_c1, arch_a1_c2, arch_a1_d0, arch_a1_d1

from board import initialBoard
from eval import evalBoard
from game import auto_player, minimax_player, simulateGames
from train import get_minmax_train_data, get_sim_games
from moves import apply_move, flip_board, get_all_moves, move_str
from minmax import minimax, minimax_eval_board

import os

from utils import onehot_encode_board, softmax
from view import print_board
# Disable GPU training:
# tf.config.set_visible_devices([], 'GPU')

# Chess board representation will be a 8x8x6 where the 3rd dimension, each one represents a piece type
# Piece types: 0 Pawn, 1 Knight,  2 Bishop, 3 Rook, 4 Queen, 5 King

print("generating test data")

# data = np.load("train_data/get_sim_games_2_16384.npz")
(x_train, y_train) = get_sim_games(minimax_eval_board(0, 1000000, evalBoard), size=10000, verbose= False)
(x_test, y_test) = get_sim_games(minimax_eval_board(0, 1000000, evalBoard), size=1024, verbose= False)

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
