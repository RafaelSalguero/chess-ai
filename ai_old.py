import tensorflow as tf
import numpy as np
from ai_arch import arch_a0, arch_a1_c0, arch_a1_c1, arch_a1_c2, arch_a1_d0, arch_a1_d1

from board import initialBoard
from eval import evalBoard
from game import auto_player, minimax_player, simulateGames
from train import get_minmax_train_data
from moves import apply_move, flip_board, get_all_moves, move_str
from minmax import minimax

import os

from utils import softmax
# Disable GPU training:
tf.config.set_visible_devices([], 'GPU')

# Chess board representation will be a 8x8x6 where the 3rd dimension, each one represents a piece type
# Piece types: 0 Pawn, 1 Knight,  2 Bishop, 3 Rook, 4 Queen, 5 King


def create_model(arch):
    model = arch()

    print(model.summary())

    loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer='adam', loss=loss_fn, metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def train_model(model, data):
    ((x_train, y_train), (x_test, y_test)) =  data

    print("train size: " + str(x_train.shape[0]))

    model.fit(x_train, y_train, epochs=1000, batch_size=32, callbacks=[
         tf.keras.callbacks.EarlyStopping("loss", min_delta=5, patience=20, mode="min")
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
def ai_player(model, verbose = False):
    def find_best(board):
        return find_best_move_ai(model, board, verbose=verbose)
    
    def player (board, color):
        return auto_player(board, color, find_best)
    
    return player

def ai_player_amplified(model, verbose = False):
    def eval(board, color):
        return ai_eval_board (model, board, color)
    return minimax_player(2, eval)

def run():
    old_model_name = "evalBoard_0"
    # Train with the initial eval function:
    
    if(os.path.exists(f'models/{old_model_name}')):
        old_model = tf.keras.models.load_model(f'models/{old_model_name}')
    else:
        old_model = create_model(arch_a0)
        initial_train(old_model, "evalBoard", 0, 10000)

    old_model = old_model
    old_player = ai_player(old_model)

    amplify_archs = [
        [arch_a1_c2],
        [arch_a1_c2],
        [arch_a1_c2],
        [arch_a1_c2],
        [arch_a1_c2],
        [arch_a1_c2],
        ]
    for i, archs in enumerate(amplify_archs):
        print(f'old model: {old_model_name}')
        print(f"amplify step {i}")

        if(len(archs) == 0):
            print(f"skipping level {i}")
            continue
        # Test if the current model performs better with ideal amplification, if not, there is no
        # gain in trying to amplify it

        old_player = ai_player(old_model)
        old_player_ideal_amplify = ai_player_amplified(old_model)
        ideal_rate = simulateGames(initialBoard, old_player_ideal_amplify, old_player, 100, False)

        print(f'ideal amplify rate: {ideal_rate}')

        if(ideal_rate < 0.6):
            print(f'ideal rate not enough')
            break

        # Try to learn an amplified version of the model using one of the given archs
        amplified_data = amplify_training_data(old_model, f'{old_model_name}_{i}')

        best_real_rate = -1
        best_model = old_model
        best_model_name = old_model
        for arch in archs:
            arch_name = arch.__name__
            print(f'Amplify learning using {arch_name}')
            curr_model = create_model(arch)
            train_model(curr_model, amplified_data)

            curr_player = ai_player(curr_model)

            print("Simulating games:")
            real_rate = simulateGames(initialBoard, curr_player, old_player, 100, False)

            print(f"real rate: {real_rate}")

            name = f'amplify_{i}_{arch_name}_rate_{round(real_rate * 100)}'
            curr_model.save(f"models/{name}")

            if(real_rate > best_real_rate):
                best_real_rate = real_rate
                best_model = curr_model
                best_model_name = name
        
        old_model = best_model
        old_model_name = best_model_name

        if(best_real_rate < 0.55):
            print(f'best_real_rate {best_real_rate} not enogh for next step')
            continue

run()
# exit()
# old_model = tf.keras.models.load_model("models/evalBoard_0")

# amplify_data = amplify_training_data(old_model, "2", 100000)
# train_model(curr_model, amplify_data)
# curr_model.save("models/amplify_test_2")
# curr_model = tf.keras.models.load_model("models/amplify_test_2")

# old_player = minimax_player(1)
# player = ai_player(curr_model) 

# simulateGames(testInitialBoard, player, old_player, 100, False)


