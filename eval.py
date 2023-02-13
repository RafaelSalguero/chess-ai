import numpy as np
from board import king

pieceValues = np.array([
    5, # rook 
    3, # knight
    3, # bishop
    9, # queen
    150, # king
    1 # pawn
])

def evalCell(cell):
    return np.dot(cell, pieceValues)

# Returns 1 if white wins, -1 if black wins, 0 if still no winner
def evalWin(board):
    return np.dot(np.sum(board, axis=(0,1)), king)

def evalBoard(board, color):
    # color doesn't matter in static eval but we still keep it to make this function
    # compatible with minimax
    
    return np.sum(np.dot(board, pieceValues))