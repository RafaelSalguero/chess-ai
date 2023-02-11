import tensorflow as tf
import numpy as np

from board import testInitialBoard
from eval import evalBoard
from game import auto_player, minimax_player, simulateGames
from train import get_minmax_train_data
from moves import apply_move, flip_board, get_all_moves, move_str
from minmax import minimax


# Chess board representation will be a 8x8x6 where the 3rd dimension, each one represents a piece type
# Piece types: 0 Pawn, 1 Knight,  2 Bishop, 3 Rook, 4 Queen, 5 King

def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(8,8,6)))
    model.add(tf.keras.layers.Conv2D(6,3, padding="same"))
    model.add(tf.keras.layers.Conv2D(5,7, padding="same"))
    model.add(tf.keras.layers.Conv2D(4,17, padding="same"))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(128, activation = "swish"))
    model.add(tf.keras.layers.Dense(8, activation = "swish"))

    model.add(tf.keras.layers.Dense(1))

    print(model.summary())

    loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer='adam', loss=loss_fn, metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def train_model(model, data):
    ((x_train, y_train), (x_test, y_test)) =  data

    print("train size: " + str(x_train.shape[0]))

    model.fit(x_train, y_train,  epochs=1000, callbacks=[
         tf.keras.callbacks.EarlyStopping("loss", min_delta=10, patience=5, mode="min")
    ])

    model.evaluate(x_test,  y_test, verbose=2)


# Evals the position for white to play
@tf.function    
def internal_model_eval(model, x):
    return model(x)
    
def ai_eval_board(model, board, color):
    if(color == -1):
        board = flip_board(board)

    return internal_model_eval(model, board.reshape((1,8,8,6))).numpy()[0][0] * color

def train_minmax_model(model, eval_func, depth):
    def minmax_eval_board(board):
            (value,) = minimax(board, 1, depth, eval_func)
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

def initial_train(model, name, depth = 0, size=50000):
    data = get_minmax_train_data("evalBoard", evalBoard, size, 250, depth)
    train_model(model, data)
    model.save(f'models/{name}_{depth}')

def amplify_training_data(model, name, size=50000):
    def eval(board, color):
        return ai_eval_board(model, board, color)
    
    return get_minmax_train_data(f'amplify_{name}', eval, size, 250, 1, True)


# creates an AI player for the given model
def ai_player(model, depth = 1, verbose = False):
    def find_best(board):
        return find_best_move_ai(model, board, verbose=verbose)
    
    def player (board, color):
        return auto_player(board, color, find_best)
    
    return player


# old_model = create_model()
curr_model = create_model()

# Train with the initial eval function:
# initial_train(old_model, "evalBoard", 0, 10000)
# curr_model = tf.keras.models.load_model("models/evalBoard_0")

# initial_train(curr_model, "evalBoard", 0, 50000)
old_model = tf.keras.models.load_model("models/evalBoard_0")

amplify_data = amplify_training_data(old_model, "2", 100000)
train_model(curr_model, amplify_data)
curr_model.save("models/amplify_test_2")
# curr_model = tf.keras.models.load_model("models/amplify_test_1")

old_player = ai_player(old_model) 
player = ai_player(curr_model) 

simulateGames(testInitialBoard, player, old_player, 100, True)


