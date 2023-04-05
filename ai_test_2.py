from ai_train import get_train_data
from eval import evalWin
from layers import calc_layers, get_layer_data
from mcts import mcts
from minmax import iterative_deepening, variation_str
from moves import allocate_moves_array, apply_move, flip_board, get_all_moves, get_all_moves_slow, move_str
from utils import onehot_encode_board
from view import parse_board, print_board
import numpy as np
import tensorflow as tf

# Disable GPU training:
tf.config.set_visible_devices([], 'GPU')


model = tf.keras.models.load_model("models/alpha_beta_3_8M_300_it")

layer_data = get_layer_data(model.layers)


@tf.function    
def internal_model_eval_no_win_check(model, x):
    return model(x)
    
def internal_model_eval(model, x):
    return from_model_space(internal_model_eval_no_win_check(model, onehot_encode_board(x).reshape((-1, 8, 8, 8))))

def from_model_space(x):
    return np.power((x - 0.5) * (2 * np.cbrt(150)), 3)


board = parse_board(
"""
8    ♞  ♝  ♛     ♝  ♞  ♜ 
7       ♟  ♟  ♟     ♟  ♟ 
6                        
5 ♕                      
4                        
3 ♙  ♚     ♙  ♙  ♙       
2 ♙     ♙           ♙  ♙ 
1 ♖     ♗     ♔     ♘  ♖ 
  a  b  c  d  e  f  g  h
"""
)

model_input = onehot_encode_board(board).reshape(-1, 8, 8, 8)

print(f"model eval: f{internal_model_eval_no_win_check(model, model_input)}")

layers_out = calc_layers(model_input.reshape(8, 8, 8), layer_data).data1d[0]
print(f"layer calc eval: f{layers_out}")

exit()
move = mcts(board, 1, model, 15000, 1, 5, True)

print_board(board)
print(move_str(move))