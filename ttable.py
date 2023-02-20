import numpy as np
from numba import njit

from utils import get_np_hash

@njit
def set_transposition_table(table, board, color, depth, eval):
    board_hash = get_np_hash(board)
    entry_hash = np.int32(board_hash ^ get_np_hash(np.array([color])))
    entry_index = np.int32(entry_hash % table.shape[0])
    
    if(depth >= table[entry_index, 2]):
        table[entry_index, 0] = board_hash
        table[entry_index, 1] = color
        table[entry_index, 2] = depth
        table[entry_index, 3] = eval

@njit
def get_transposition_table(table, board, color, depth):
    board_hash = get_np_hash(board)
    entry_hash = board_hash ^ get_np_hash(np.array([color]))
    entry_index = entry_hash % table.shape[0]

    if(
        board_hash == table[entry_index, 0] and 
        table[entry_index, 1] == color and
        table[entry_index, 2] >= depth
        ):
        value = table[entry_index, 3]

        return (True, value) 
    
    return (False, 0)

@njit
def init_transposition_table(size):
    return np.zeros((size, 4), dtype=np.int32)