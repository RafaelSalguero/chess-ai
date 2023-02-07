import tensorflow as tf
import numpy as np
from board import get_random_board, initialBoard, pawn, piecesRank
from eval import evalBoard
from view import print_piece, print_rank


# Chess board representation will be a 8x8x6 where the 3rd dimension, each one represents a piece type
# Piece types: 0 Pawn, 1 Knight,  2 Bishop, 3 Rook, 4 Queen, 5 King


training_set_size = 10000
x_train = np.array(list(map(lambda x: get_random_board(), range(0, training_set_size))))
y_train = np.array(list(map(evalBoard, x_train)))

print_rank(piecesRank, True)
exit()

print(y_train)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(8,8,6)),
    tf.keras.layers.Conv2D(1, 1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1)
])

loss_fn = tf.keras.losses.MeanSquaredError()
model.compile(optimizer="adam", loss=loss_fn, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100)


x_test = np.array(list(map(lambda x: get_random_board(), range(0, training_set_size))))
y_test = np.array(list(map(evalBoard, x_test)))
model.evaluate(x_test,  y_test, verbose=2)
