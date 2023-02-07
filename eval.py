import numpy as np

pieceValues = np.array([
    5, # rook 
    3, # knight
    3, # bishop
    9, # queen
    50, # king
    1, # pawn
])

def evalCell(cell):
    return np.dot(cell, pieceValues)

def evalBoard(board):
    return np.sum(np.dot(board, pieceValues))