import numpy as np
from board import king, king_moved, emptyCell, knight, bishop
from numba import njit

pieceValues = np.array([
    5, # rook 
    5, # rook_moved
    3, # knight
    3, # bishop
    9, # queen
    100, # king
    100, # king_moved
    1 # pawn
], dtype=np.float32)

# Any eval greater than this indicates a win
win_value_heuristic = 50

@njit
def evalCell(cell):
    return np.dot(cell, pieceValues)

@njit
def evalDeadPosition(board):
    """
        Returns true if the board is a dead position by insufficient material
    """

    a_sum = 0
    b_sum = 0 
    for y in range(0, 8):
        for x in range(0, 8):
            signed_piece = board[y, x]
            a = abs(signed_piece)
            if(a == emptyCell or a == king or a == king_moved): 
                continue
            if(
                a != knight and
                a != bishop):
                return False
            
            square_color = (x + y) % 2
            piece_val = (
                4 if a == knight else 
                2 if a == bishop and square_color == 1 else 
                1
            )
            if(signed_piece > 0):
                a_sum += piece_val
            else:
                b_sum += piece_val
    
    # a is always greater or equal
    if(a_sum < b_sum):
        (b_sum, a_sum) = (a_sum, b_sum)

    return (
        (a_sum == 0 and b_sum == 0) or # king vs king
        (a_sum == 4 and b_sum == 0) or # knight vs king
        (a_sum == 2 and b_sum == 0) or # king and bishop vs king (white bishop)
        (a_sum == 1 and b_sum == 0) or # king and bishop vs king (black bishop)
        (a_sum == b_sum and (a_sum == 2 or a_sum == 1)) # king and bishop vs king and bishop of the same color 
    )


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