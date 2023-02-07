import tensorflow as tf
import numpy as np
import itertools as it

def piece(index):
    ret = [0,0,0,0,0,0]
    ret[index] = 1
    return np.array(ret)

emptyCell = np.zeros(6)
rook = piece(0)
knight = piece(1)
bishop = piece(2)
queen = piece(3)
king = piece(4)
pawn = piece(5)



piecesRank = np.array([rook, knight, bishop, queen, king, bishop, knight, rook ])
pawnsRank = np.array([pawn, pawn, pawn, pawn, pawn, pawn, pawn, pawn])
emptyRank = np.array([emptyCell, emptyCell, emptyCell, emptyCell, emptyCell, emptyCell, emptyCell, emptyCell])

initialBoard = np.array([
    -piecesRank,
    -pawnsRank, 
    emptyRank,
    emptyRank,
    emptyRank,
    emptyRank,
    pawnsRank,
    piecesRank
])


def get_random_board():
    # Generates a random chess board
    all_coords = np.arange(0, 64, 1)
    all_pieces = np.array([ -piecesRank, -pawnsRank, pawnsRank, piecesRank ]).reshape(-1, 6)
    np.random.shuffle(all_pieces)
    np.random.shuffle(all_coords)
    piece_count = np.random.randint(all_pieces.shape[0])
    random_board = np.zeros((64, 6))
    np.add.at(random_board,all_coords[0: piece_count], all_pieces[0: piece_count])
    return random_board.reshape(8, 8, 6)


def printPiece(piece):
    chars = ['♜', '♞', '♝', '♛', '♚', '♟︎']