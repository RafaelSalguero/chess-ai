import numpy as np
from ai_old import internal_model_eval

from moves import apply_move, get_all_moves



# Finds the best move for white
def ai_find_best_move(board):
    moves = get_all_moves(board, 1)
    boards = np.array(list(map(lambda move: apply_move(board, move), moves)))
    
    evals = internal_model_eval(boards)
    return moves[np.argmax(evals)]