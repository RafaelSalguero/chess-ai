import numpy as np
from numba import njit
from moves import flip_board

from utils import get_np_hash


depth_mul = 10000
depth_bias = 1
value_bias = 1000
@njit
def set_transposition_table(table, board, color, depth, eval):
    board_hash = get_np_hash(board if color == 1 else flip_board(board))
    entry_index = np.int32(board_hash % table.shape[0])
    
    table_depth = (table[entry_index] // depth_mul) - depth_bias
    if(depth >= table_depth):
        table[entry_index] = (depth + depth_bias) * depth_mul + (eval * color + value_bias)

@njit
def get_transposition_table(table, board, color, depth):
    board_hash = get_np_hash(board if color == 1 else flip_board(board))
    entry_index = board_hash % table.shape[0]

    table_depth = (table[entry_index] //depth_mul) - depth_bias
    if(table[entry_index] != 0 and table_depth >= depth):
        value = ((table[entry_index] % depth_mul) - value_bias) * color
        return (True, value) 
    
    return (False, 0)

@njit
def init_transposition_table(size):
    return np.zeros(size, dtype=np.int32)