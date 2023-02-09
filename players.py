import tensorflow as tf
import numpy as np
from ai import internal_model_eval

from board import get_test_random_board, get_random_board, initialBoard, pawn, piecesRank
from eval import evalBoard, win_threshold
from train import get_minmax_train_data
from view import print_board, print_piece, print_rank
from moves import apply_move, flip_board, flip_move, get_all_moves, move_str, str_move
from minmax import minimax



# Finds the best move for white
def ai_find_best_move(board):
    moves = get_all_moves(board, 1)
    boards = np.array(list(map(lambda move: apply_move(board, move), moves)))
    
    evals = internal_model_eval(boards)
    return moves[np.argmax(evals)]