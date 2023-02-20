import numpy as np
from numba import njit
from moves import flip_board

from utils import get_np_hash


depth_mul = 10000
@njit
def set_transposition_table(table, board, color, depth, eval):
    board_hash = get_np_hash(board if color == 1 else flip_board(board))
    entry_index = np.int32(board_hash % table.shape[0])
    
    table_depth = (table[entry_index] // depth_mul) - 1
    if(depth >= table_depth):
        table[entry_index] = (depth + 1) * 10000 + eval * color

@njit
def get_transposition_table(table, board, color, depth):
    board_hash = get_np_hash(board if color == 1 else flip_board(board))
    entry_index = board_hash % table.shape[0]

    table_depth = (table[entry_index] //depth_mul) - 1
    if(table[entry_index] != 0 and table_depth >= depth):
        value = (table[entry_index] % depth_mul) * color
        return (True, value) 
    
    return (False, 0)

@njit
def init_transposition_table(size):
    return np.zeros(size, dtype=np.int32)