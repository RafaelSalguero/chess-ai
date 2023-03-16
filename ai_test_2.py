from eval import evalWin
from minmax import iterative_deepening
from moves import allocate_moves_array, apply_move, flip_board, get_all_moves, get_all_moves_slow
from utils import onehot_encode_board
from view import parse_board, print_board
import numpy as np
import tensorflow as tf

# Disable GPU training:
tf.config.set_visible_devices([], 'GPU')


model = tf.keras.models.load_model("models/alpha_beta_3_10M")

@tf.function    
def internal_model_eval_no_win_check(model, x):
    return model(x)
    
def internal_model_eval(model, x):
    win = evalWin(x)
    if(win != 0):
        return np.array([[win * 1000]])
    else:
        return from_model_space(internal_model_eval_no_win_check(model, onehot_encode_board(x).reshape((-1, 8, 8, 8))))

def from_model_space(x):
    return np.power((x - 0.5) * (2 * np.cbrt(150)), 3)


board = parse_board(
"""
8 ♜  ♞  ♝  ♛  ♚  ♝  ♞  ♜ 
7                ♟  ♟  ♟ 
6                        
5 ♟  ♙                   
4 ♙     ♙                
3          ♟  ♙  ♙     ♙ 
2       ♖  ♕  ♔     ♙    
1    ♘  ♗           ♘  ♖ 
  a  b  c  d  e  f  g  h
"""
)

all_moves = get_all_moves_slow(board, -1)
next_boards = list(map(lambda move: flip_board(apply_move(board, move)), all_moves))

for next_board in next_boards:
    print_board(next_board)
    print(internal_model_eval(model, next_board))
    (value, _, _, _) = iterative_deepening(2, 2, 100, next_board, 1, None, None, allocate_moves_array(), 0)

    print(value)