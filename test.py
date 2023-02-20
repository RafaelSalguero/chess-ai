import random
from moves import get_all_moves, apply_move, flip_board
from math import ceil

# Returns an aprox number of test boards
def get_test_boards(board, color, depth, count, ret = []):
    if(count <=0):
        return

    moves = get_all_moves(board, color)

    if(len(moves) == 0):
        return

    moves_per_depth = count / depth
    prob_of_explore = min(moves_per_depth / len(moves), 1)
    
    actual_move_count = ceil(prob_of_explore * len(moves))
    random.shuffle(moves)
    actual_moves = moves[0:actual_move_count]

    if(len(actual_moves) == 0):
        return
    sub_count = ceil(max(count - actual_move_count, 0) / len(actual_moves))
    for move in actual_moves:
        next_board = apply_move(board, move)
        ret.append(next_board)
        ret.append(flip_board(next_board))

        get_test_boards(next_board, -color, max(depth - 1, 1),sub_count, ret)
    
    return ret