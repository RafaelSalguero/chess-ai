import tensorflow as tf
import numpy as np

from board import get_test_random_board, get_random_board, initialBoard, pawn, piecesRank
from eval import evalBoard, win_threshold
from train import get_minmax_train_data
from view import print_board, print_piece, print_rank
from moves import apply_move, flip_board, flip_move, get_all_moves, move_str, str_move
from minmax import minimax


# Chess board representation will be a 8x8x6 where the 3rd dimension, each one represents a piece type
# Piece types: 0 Pawn, 1 Knight,  2 Bishop, 3 Rook, 4 Queen, 5 King

model = tf.keras.Sequential([
    tf.keras.Input(shape=(8,8,6)),
    tf.keras.layers.Conv2D(8,1, activation="relu"),
    tf.keras.layers.Conv2D(4,3, activation="relu", padding="same"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1),
])

print(model.summary())

loss_fn = tf.keras.losses.MeanSquaredError()
model.compile(optimizer='adam', loss=loss_fn, metrics=[tf.keras.metrics.RootMeanSquaredError()])

def train_model(data):
    ((x_train, y_train), (x_test, y_test)) =  data

    print("train size: " + str(x_train.shape[0]))

    for i in range(0,20):
        print_board(x_train[i])
        print("y_train: " + str(y_train[i]))

    model.fit(x_train, y_train,  epochs=1000, callbacks=[
         tf.keras.callbacks.EarlyStopping("loss", min_delta=5, patience=10, mode="min")
    ])

    model.evaluate(x_test,  y_test, verbose=2)


@tf.function    
def internal_model_eval(x):
    return model(x)
    
def ai_eval_board(board):
    return internal_model_eval(board.reshape((1,8,8,6))).numpy()[0][0]

def train_minmax_model(eval_func, depth):
    def minmax_eval_board(board):
            return minimax(board, 1, depth, eval_func, win_threshold)
    train_model(minmax_eval_board)
     

# Finds the best move for white
def find_best_move(board):
    moves = get_all_moves(board, 1)
    boards = np.array(list(map(lambda move: apply_move(board, move), moves)))
    
    evals = internal_model_eval(boards)
    return moves[np.argmax(evals)]

data = get_minmax_train_data("evalBoard", get_test_random_board, evalBoard, 10000, 250, 0)
train_model(data)