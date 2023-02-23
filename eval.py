import numpy as np
from board import king, king_moved
from numba import njit

pieceValues = np.array([
    5, # rook 
    5, # rook_moved
    3, # knight
    3, # bishop
    9, # queen
    300, # king
    300, # king_moved
    1 # pawn
], dtype=np.float32)

# Any value above this represents a win
win_value = 300 - (9 * 9 + (5 + 3 + 3) * 2) + 1

@njit
def evalCell(cell):
    return np.dot(cell, pieceValues)

# Returns 1 if white wins, -1 if black wins, 0 if still no winner
@njit
def evalWin(board):
    s = np.sign(board)
    a = s * board
    return np.sum((a == king) * s) + np.sum((a == king_moved) * s)

@njit
def evalBoard(board, color):
    """Evaluates the board from the perspective of the given color. Example, color 1 is positive for a better position for white, color -1 is positive for a better position for black
    """
    s = np.sign(board)
    a = s * board

    return np.sum(((a == 1) * 5 + (a == 2) * 5 + (a == 3) * 3 + (a == 4) * 3 + (a == 5) * 9 + (a == 6) * 150 + (a == 7) * 150 + (a == 8) * 1) * s) * color