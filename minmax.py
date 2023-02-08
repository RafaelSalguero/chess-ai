import numpy as np
from moves import get_all_moves, apply_move, move_str, flip_board


# Evals the position for white
def minimax(board, depth, eval_func, win_threshold):
    eval = eval_func(board)
    if(depth == 0 or abs(eval) >= win_threshold):
        return eval
    
    
    moves = get_all_moves(board)
    if(len(moves)==0): 
        return eval

    value = -10000
    for move in moves:
        nextBoard = flip_board(apply_move(board, move))

        value = max(value, -minimax(nextBoard, depth - 1, eval_func, win_threshold))
    
    return value