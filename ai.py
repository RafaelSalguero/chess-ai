import tensorflow as tf
import numpy as np

from board import testInitialBoard
from eval import evalBoard, win_threshold
from game import auto_player, simulateGames
from train import get_minmax_train_data
from view import print_board
from moves import apply_move, flip_board, get_all_moves, move_str
from minmax import minimax


# Chess board representation will be a 8x8x6 where the 3rd dimension, each one represents a piece type
# Piece types: 0 Pawn, 1 Knight,  2 Bishop, 3 Rook, 4 Queen, 5 King

def create_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(8,8,6)),
        tf.keras.layers.Conv2D(6, 5, 1, padding="same", activation="swish"),
        tf.keras.layers.Conv2D(2, 3, 1, padding="same", activation="swish"),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.Dense(32, activation="swish"),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1),
    ])

    print(model.summary())

    loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer='adam', loss=loss_fn, metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def train_model(model, data):
    ((x_train, y_train), (x_test, y_test)) =  data

    print("train size: " + str(x_train.shape[0]))

    for i in range(0,20):
        print_board(x_train[i])
        print("y_train: " + str(y_train[i]))

    model.fit(x_train, y_train,  epochs=100, callbacks=[
         tf.keras.callbacks.EarlyStopping("loss", min_delta=5, patience=10, mode="min")
    ])

    model.evaluate(x_test,  y_test, verbose=2)


@tf.function    
def internal_model_eval(model, x):
    return model(x)
    
def ai_eval_board(model, board):
    return internal_model_eval(model, board.reshape((1,8,8,6))).numpy()[0][0]

def train_minmax_model(model, eval_func, depth):
    def minmax_eval_board(board):
            (value,) = minimax(board, 1, depth, eval_func, win_threshold)
            return value
    train_model(model, minmax_eval_board)
     
def softmax(x):
    return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())

# Finds the best move for white
def find_best_move_ai(model, board, verbose = False):
    moves = get_all_moves(board, 1)
    boards = np.array(list(map(lambda move: flip_board(apply_move(board, move)), moves)))
    
    evals = -internal_model_eval(model, boards).numpy().reshape(-1)

    probs = softmax(evals)

    best = np.random.choice(len(moves), size=1, p = probs)[0]
    
    movess = list(map(move_str, moves))
    if(verbose):
        print("moves: ", movess, "evals", evals, "probs", probs, "best", best)
        
    return moves[best]

def initial_train(model, name, depth = 0, size=100000):
    data = get_minmax_train_data("evalBoard", evalBoard, size, 250, depth)
    train_model(model, data)
    model.save(f'models/{name}_{depth}')


# creates an AI player for the given model
def ai_player(model, verbose = False):
    def find_best(board):
        return find_best_move_ai(model, board, verbose=verbose)
    
    def player (board, color):
        return auto_player(board, color, find_best)
    
    return player


# train with 0 data:


old_model = tf.keras.models.load_model('models/evalBoard_0')

curr_model = tf.keras.models.load_model('models/evalBoard_1')
initial_train(curr_model,'evalBoard', 1, 50000)

old_player = ai_player(curr_model)
player = ai_player(old_model)

simulateGames(testInitialBoard, old_player, player, 50, True)
