import numpy as np
import tensorflow as tf
from eval import evalBoard, evalWin
from game import auto_player, minimax_player, play, simulateGames

from board import initialBoard
from mcts import mcts
from moves import apply_move, flip_board, get_all_moves, get_all_moves_slow, move_str
from utils import onehot_encode_board, softmax

# Disable GPU training:
tf.config.set_visible_devices([], 'GPU')

model = tf.keras.models.load_model("models/alpha_beta_3_8M_300_it")

@tf.function    
def internal_model_eval_no_win_check(model, x):
    return model(x)
    
def internal_model_eval(model, x):
   return from_model_space(internal_model_eval_no_win_check(model, onehot_encode_board(x).reshape((-1, 8, 8, 8))).numpy().reshape(-1))

def from_model_space(x):
    return np.power((x - 0.5) * (2 * np.cbrt(150)), 3)


# Finds the best move for white
def find_best_move_ai(model, board, verbose = False):
    moves = get_all_moves_slow(board, 1)
    boards = np.array(list(map(lambda move: flip_board(apply_move(board, move)), moves)))
    
    evals = -internal_model_eval(model, boards)

    probs = softmax(evals)

    best = np.random.choice(len(moves), size=1, p = probs)[0]
    
    movess = list(map(move_str, moves))
    if(verbose):
        print("moves: ", movess, "evals", evals, "probs", probs, "best", best)
        
    return moves[best]


# creates an AI player for the given model
def ai_player(model, verbose = False):
    def find_best(board):
        return find_best_move_ai(model, board, verbose=verbose)
    
    def player (board, color):
        return auto_player(board, color, find_best)
    
    return player

def sim_player(model, verbose = False):
    def find_best(board):
        return mcts(board, 1, model, 15000, 0.8, 5)
    
    def player (board, color):
        return auto_player(board, color, find_best)
    
    return player

win_rate = simulateGames(initialBoard, sim_player(model, False), minimax_player(4), 20, True)
print("win_rate: ", win_rate)