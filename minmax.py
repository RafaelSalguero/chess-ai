import numpy as np
from board import get_test_random_board
from moves import get_all_moves, apply_move, move_str, flip_board
from view import print_board
from eval import evalBoard, win_threshold

inf_val = 100000
def alphabeta(board, depth, alpha, beta, eval_func, win_threshold):
    eval = eval_func(board)
    if(depth == 0 or abs(eval) >= win_threshold):
        return eval
    
    moves = get_all_moves(board)
    if(len(moves)==0): 
        return eval
    
    value = -inf_val

    for move in moves:
        nextBoard = flip_board(apply_move(board, move))
        value = max(value, -alphabeta(nextBoard, depth - 1, -beta, -alpha, eval_func, win_threshold))

        if value > beta:
            break

        alpha = max(alpha, value)

    return value

def minimax(board, depth, eval_func, win_threshold):
    return alphabeta(board, depth, -inf_val, inf_val, eval_func, win_threshold)