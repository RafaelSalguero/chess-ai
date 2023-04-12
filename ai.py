from ai_train import get_train_data, reduce_non_wins
from minmax import alphabeta, iterative_deepening, inf_val, minimax
import tensorflow as tf

import numpy as np
from eval import evalBoard, evalWin

from train import get_sim_games

import os

from utils import cap_histogram, onehot_encode_board
from view import print_board
# Disable GPU training:
tf.config.set_visible_devices([], 'GPU')

# Chess board representation will be a 8x8x6 where the 3rd dimension, each one represents a piece type
# Piece types: 0 Pawn, 1 Knight,  2 Bishop, 3 Rook, 4 Queen, 5 King




# data = np.load("train_data/get_sim_games_2_16384.npz")
print("generating train data")
(x_train, y_train, ply_train) = get_train_data(0, 0, 1000, 1000000, True)

print("generating test data")

(x_test, y_test, ply_test) = get_train_data(0, 0, 1000, 500000, True)

if(True):
    reduce_non_wins(x_train, y_train)
    reduce_non_wins(x_test, y_test)

if(True):
    # cap histogram
    indices = cap_histogram(y_train, 200000, 1)

    print(f"original len: {len(y_train)}")
    x_train = x_train[indices]
    y_train = y_train[indices]

    print(f"capped len: {len(y_train)}")

    bins = np.arange(-150, 152, 1)
    (h, bins) = np.histogram(y_train, bins)
    print(h)
# exit()


print(f"train sample {x_train.shape[0]} {y_train.shape[0]}:")

def from_model_space(x):
    return np.pow((x - 0.5) * (2 * np.cbrt(150)), 3)

def from_model_space_tf(x):
    return tf.math.pow((x - 0.5) * (10.6265), 3)

def to_model_space(x):
    """
        Maps from [-150, 150] to [0, 1]
    """
    return np.cbrt(x) / (2 * np.cbrt(150)) + 0.5

print("train sample:")
for i in np.random.random_integers(0, x_train.shape[0] - 1, 20):
    print_board(x_train[i])
    print(f"eval: {y_train[i]}")

print("test sample:")

for i in np.random.random_integers(0, x_test.shape[0] - 1, 20):
    print_board(x_test[i])
    print(f"eval: {y_test[i]}")

model_x_test = onehot_encode_board(x_test)
model_x_train = onehot_encode_board(x_train)

model_y_train = to_model_space(y_train)
model_y_test = to_model_space(y_test)

def create_model():
    inputs = tf.keras.Input(shape=(8,8,8))

    x = inputs

    x = tf.keras.layers.Conv2D(8, 1, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(2, 1, padding="same", activation="relu")(x)

    x = tf.keras.layers.AveragePooling2D(pool_size=(8,8))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4, activation = "relu")(x)
    x = tf.keras.layers.Dense(1, activation = "sigmoid")(x)

    model = tf.keras.Model(inputs = inputs, outputs = x)
    return model


model = create_model()


print(model.summary())

model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])

print("train size: " + str(x_train.shape[0]))

model.fit(model_x_train, model_y_train, validation_data = (model_x_test, model_y_test), epochs=50, batch_size=64)

model.save("models/alpha_beta_3")

model.evaluate(model_x_train,  model_y_train, verbose=2)
model.evaluate(model_x_test,  model_y_test, verbose=2)

#y_eval = model(model_x_test)
#print("eval sample:")
#for i in np.random.random_integers(0, x_test.shape[0] - 1, 100):
#    print_board(x_test[i])
#    print(f"test: {y_test[i]} evalBoard: {evalBoard(x_test[i], 1)} value: {y_eval[i]}")

