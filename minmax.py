from moves import get_all_moves, move_str, apply_move_inplace, undo_move_inplace
from eval import evalBoard
from numba import njit
from ttable import get_transposition_table, init_transposition_table, set_transposition_table
from utils import get_np_hash
import numpy as np
class Variation:
    def __init__(self, parent, value):
        self.parent = parent
        self.value = value

def variation_str(variation):
    if(variation == None):
        return ""
    if(variation.parent == None and variation.value == None):
        return "-"
    return variation_str(variation.parent) + '->' + move_str(variation.value)



inf_val = 100000
@njit
def alphabeta(board, color, depth, alpha, beta, eval_func, parent_move, ttable):
    iters = 1
    
    if(ttable != None):
        (ttable_hit, ttable_read) = get_transposition_table(ttable, board, color, depth)
        if(ttable_hit):
            return (ttable_read, parent_move, None, iters)

    if(depth == 0):
        eval = eval_func(board, color)
        if(ttable != None):
            set_transposition_table(ttable, board, color, depth, eval)
        return (eval, parent_move, None, iters)
    
    moves = get_all_moves(board, color)
    if(len(moves)==0): 
        eval = eval_func(board, color)
        if(ttable != None):
            set_transposition_table(ttable, board, color, depth, eval)
        return (eval, parent_move, None, iters)
    
    value = -inf_val

    best_variation = None
    best_move = None

    #random.shuffle(moves)
    for move in moves:
        undo = apply_move_inplace(board, move)

        curr_variation = Variation(parent_move, move) if parent_move != None else None
        (move_eval, next_variation, _, child_iters) = alphabeta(board, -color, depth - 1, -beta, -alpha, eval_func, curr_variation, ttable)
        iters += child_iters
        move_eval = -move_eval

        undo_move_inplace(board, move, undo)

        if(move_eval > value):
            value = move_eval
            best_move = move
            best_variation = next_variation

        if value > beta:
            break

        alpha = max(alpha, value)

    if(ttable != None):
        set_transposition_table(ttable, board, color, depth, value)
        
    return (value, best_variation, best_move, iters)

def minimax(board, color, depth, eval_func, calc_variation = False):
    return alphabeta(board, color, depth, -inf_val, inf_val, eval_func, Variation(None, None) if calc_variation else None)

@njit
def iterative_deepening(max_depth, max_iter, board, color, ttable):
    value = 0
    rem_iters = max_iter
    for depth in range(0, max_depth + 1):
        (value, _, _, iters) = alphabeta(board, color, depth, -inf_val, inf_val, evalBoard, None, ttable)
        rem_iters -= iters
        if(rem_iters <= 0):
            break
        
    return value

def find_best_move_minimax(board, depth, eval_func):
    if(depth < 1):
        raise Exception("Depth should be >= 1")
    
    (value, best_variation, best_move) = minimax(board, 1, depth, eval_func, False)
    return best_move