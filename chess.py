import tensorflow as tf
import numpy as np

from board import get_test_random_board, get_random_board, initialBoard, pawn, piecesRank
from eval import evalBoard, win_threshold
from view import print_board, print_piece, print_rank
from moves import apply_move, flip_board, flip_move, get_all_moves, move_str, str_move
from minmax import minimax


# Chess board representation will be a 8x8x6 where the 3rd dimension, each one represents a piece type
# Piece types: 0 Pawn, 1 Knight,  2 Bishop, 3 Rook, 4 Queen, 5 King

model = tf.keras.Sequential([
    tf.keras.Input(shape=(8,8,6)),
    tf.keras.layers.Conv2D(8,1, activation="relu"),
    tf.keras.layers.Conv2D(3,3, activation="relu", padding="same"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1),
])

print(model.summary())

loss_fn = tf.keras.losses.MeanSquaredError()
model.compile(optimizer='adam', loss=loss_fn, metrics=[tf.keras.metrics.RootMeanSquaredError()])

def get_train_data(eval_func, size):
    x_train = np.array(list(map(lambda x: get_test_random_board(), range(0, size))))
    y_train = np.array(list(map(eval_func, x_train)))
    return (x_train, y_train)

def train_model(eval_func):
    print("generating test data...")
    training_set_size = 10000
    (x_train, y_train) = get_train_data(eval_func, training_set_size)

    for i in range(0,20):
        print_board(x_train[i])
        print("y_train: " + str(y_train[i]))

    model.fit(x_train, y_train,  epochs=1000, callbacks=[
         tf.keras.callbacks.EarlyStopping("loss", min_delta=10, patience=3, mode="min")
    ])

    validate_test_size = 250
    (x_test, y_test) = get_train_data(eval_func, validate_test_size)
    model.evaluate(x_test,  y_test, verbose=2)


@tf.function    
def internal_model_eval(x):
    return model(x)
    
def ai_eval_board(board):
    return internal_model_eval(board.reshape((1,8,8,6))).numpy()[0][0]

def train_minmax_model(eval_func, depth):
    def minmax_eval_board(board):
            return minimax(board, depth, eval_func, win_threshold)
    train_model(minmax_eval_board)
     

# Finds the best move for white
def find_best_move(board):
    moves = get_all_moves(board)
    boards = np.array(list(map(lambda move: apply_move(board, move), moves)))
    
    evals = internal_model_eval(boards)
    
    print("model eval: " + str(np.max(evals)))
    return moves[np.argmax(evals)]

train_minmax_model(evalBoard, 0)