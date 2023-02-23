from moves import apply_move, get_all_moves, is_empty_cell, move_str, apply_move_inplace, move_str_an, undo_move_inplace
from eval import evalBoard, win_value
from numba import njit, deferred_type, int32, optional, typeof
from numba.experimental import jitclass
from ttable import get_transposition_table, init_transposition_table, set_transposition_table
from utils import get_np_hash, shuffle
from collections import OrderedDict

import numpy as np

node_type = deferred_type()

spec = OrderedDict()
spec['value'] = typeof((np.array([0,0]), np.array([0,0])))
spec['parent'] = optional(node_type)

@jitclass(spec)
class Variation:
    def __init__(self, parent, value):
        self.parent = parent
        self.value = value

node_type.define(Variation.class_type.instance_type)

@njit
def variation_str(variation):
    if(variation is None):
        return ""
    if(variation.parent is None):
        return "-"
    return variation_str(variation.parent) + '->' + move_str(variation.value)

@njit
def variation_str_an(variation, board, color):
    if(variation.parent is None):
        return ("-", board, color)
    
    (parent_str, board, color) = variation_str_an(variation.parent, board, color)

    curr_str = move_str_an(board, get_all_moves(board, color), variation.value)

    board = apply_move(board, variation.value)
    color = -color

    return (parent_str + '->' + curr_str, board, color)


inf_val = 300

@njit
def alphabeta(board, color, quiescence, depth, quiescence_depth, max_iter, alpha, beta, eval_func, parent_move, ttable, ttable_write):
    iters = 1
    
    if(ttable is not None):
        (ttable_hit, ttable_read) = get_transposition_table(ttable, board, color, depth)
        if(ttable_hit):
            return (ttable_read, parent_move, None, iters)

    if(depth == 0):
        eval = eval_func(board, color)
        if(ttable is not None and ttable_write):
            set_transposition_table(ttable, board, color, depth, eval)
        return (eval, parent_move, None, iters)
    
    moves = get_all_moves(board, color)
    
    value = -inf_val

    best_variation = None
    best_move = None

    #random.shuffle(moves)
    shuffle(moves)
    move_count = 0
    for move in moves:
        is_take = not is_empty_cell(board, move[1])
        if(quiescence and not is_take):
            continue
        move_count += 1

        undo = apply_move_inplace(board, move)

        curr_variation = Variation(parent_move, move) if parent_move != None else None

        next_depth = depth - 1
        next_quiescence = quiescence
        if(is_take and depth == 1 and not quiescence and quiescence_depth > 0):
            # enable quiescence search:
            next_quiescence = True
            next_depth = quiescence_depth
        
        next_max_iter = max_iter - iters
        (move_eval, next_variation, _, child_iters) = alphabeta(board, -color, next_quiescence, next_depth, quiescence_depth, next_max_iter, -beta, -alpha, eval_func, curr_variation, ttable, ttable_write and not quiescence)
        iters += child_iters

        undo_move_inplace(board, move, undo)

        if(iters > max_iter or child_iters == -1):
            # early break:
            return (value, best_variation, best_move, -1)

        move_eval = -move_eval


        if(move_eval > value):
            value = move_eval
            best_move = move
            best_variation = next_variation

        if value > beta:
            break

        alpha = max(alpha, value)

    if(move_count==0): 
        eval = eval_func(board, color)
        if(ttable is not None and ttable_write):
            set_transposition_table(ttable, board, color, depth, eval)
        return (eval, parent_move, None, iters)
    
    # TODO Check for win/loss instead of win_value
    if(quiescence and value < -win_value):
        # If the game was lost on quiescence search, restart with normal search:
        print("quiescence check {value}")
        return alphabeta(board, color, False, 1, quiescence_depth, max_iter - iters, alpha, beta, eval_func, parent_move, ttable, ttable_write)

    if(ttable is not None and ttable_write):
        set_transposition_table(ttable, board, color, depth, value)
        
    return (value, best_variation, best_move, iters)

def minimax(board, color, depth, eval_func, calc_variation = False):
    return alphabeta(board, color, depth, -inf_val, inf_val, eval_func, Variation(None, None) if calc_variation else None)

@njit
def iterative_deepening(max_depth, quiescence_depth, max_iter, board, color, ttable, calc_variation):
    value = 0
    rem_iters = max_iter

    root_variation = Variation(None, (np.array([0,0]), np.array([0,0])))
    best_variation = root_variation
    best_depth = 0
    for depth in range(0, max_depth + 1):
        (next_value, next_best_variation, _, iters) = alphabeta(board, color, False, depth, quiescence_depth, rem_iters, -inf_val, inf_val, evalBoard, root_variation if calc_variation else None, ttable, True)
        if(depth > 0 and iters  == -1):
            # early break, this result is not valid, we take the previous one
            break

        value = next_value
        best_variation = next_best_variation
        best_depth = depth
        rem_iters -= iters
        
    return (value, best_variation, max_iter - rem_iters, best_depth)

def find_best_move_minimax(board, depth, eval_func):
    if(depth < 1):
        raise Exception("Depth should be >= 1")
    
    (value, best_variation, best_move) = minimax(board, 1, depth, eval_func, False)
    return best_move