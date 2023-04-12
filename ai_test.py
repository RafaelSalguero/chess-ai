import numpy as np
import tensorflow as tf
from eval import evalBoard, evalWin
from game import auto_player, minimax_player, play, simulateGames

from board import initialBoard
from layers import get_layer_data
from mcts_fast import mcts
from moves import apply_move, flip_board, get_all_moves, get_all_moves_slow, move_str
from ttable import init_transposition_table
from utils import onehot_encode_board, softmax
from view import parse_board

# Disable GPU training:
tf.config.set_visible_devices([], 'GPU')

model = tf.keras.models.load_model("models/alpha_beta_3_small")
layer_data = get_layer_data(model.layers)

def sim_player(model, verbose = False):
    rep_table = init_transposition_table(16384)
    eval_ttable = init_transposition_table(1024 * 1024 * 1024)
    def find_best(board):
        return mcts(board, 200000, rep_table, eval_ttable, layer_data, 1, 3, False)
    
    def player (board, color):
        return auto_player(board, color, find_best)
    
    return player

board = parse_board(
"""
8                        
7    ♚     ♔        ♗    
6                        
5                ♕       
4                        
3                        
2                        
1                      ♖ 
  a  b  c  d  e  f  g  h
"""
)
win_rate = simulateGames(initialBoard, sim_player(model, False), minimax_player(4), 1, True)
print("win_rate: ", win_rate)