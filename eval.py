import numpy as np
from board import king

pieceValues = np.array([
    5, # rook 
    3, # knight
    3, # bishop
    9, # queen
    300, # king
    1, # pawn
])

# Any eval greater than 150 is considered a win
win_threshold = 150

def evalCell(cell):
    return np.dot(cell, pieceValues)

# Returns 1 if white wins, -1 if black wins, 0 if still no winner
def evalWin(board):
    return np.dot(np.sum(board, axis=(0,1)), king)

def evalBoard(board):
    return np.sum(np.dot(board, pieceValues))