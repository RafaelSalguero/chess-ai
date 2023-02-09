import numpy as np
from board import get_test_random_board
from moves import flip_move, get_all_moves, apply_move, move_str, flip_board, apply_move_inplace, undo_move_inplace
from view import print_board
from eval import evalBoard, win_threshold

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
def alphabeta(board, color, depth, alpha, beta, eval_func, win_threshold, parent_move):
    eval = eval_func(board) * color
    if(depth == 0 or abs(eval) >= win_threshold):
        return (eval, parent_move)
    
    moves = get_all_moves(board, color)
    if(len(moves)==0): 
        return (eval, parent_move)
    
    value = -inf_val

    best_variation = None
    for move in moves:
        undo = apply_move_inplace(board, move)

        curr_variation = Variation(parent_move, move) if parent_move != None else None
        (move_eval, next_variation) = alphabeta(board, -color, depth - 1, -beta, -alpha, eval_func, win_threshold, curr_variation)
        move_eval = -move_eval

        undo_move_inplace(board, move, undo)

        if(move_eval > value):
            value = move_eval
            best_variation = next_variation

        if value > beta:
            break

        alpha = max(alpha, value)

    return (value, best_variation)

def minimax(board, color, depth, eval_func, win_threshold):
    (value, best_variation) = alphabeta(board, color, depth, -inf_val, inf_val, eval_func, win_threshold, Variation(None, None))

    # print(variation_str(best_variation))
    return value * color