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

testInitialBoard_old = np.array([
    -np.array([ emptyCell, emptyCell, emptyCell, emptyCell, emptyCell, emptyCell, emptyCell, king]),
    -np.array([ emptyCell, emptyCell, emptyCell, emptyCell, emptyCell, emptyCell, pawn, emptyCell]),
    emptyRank,
    emptyRank,
    emptyRank,
    emptyRank,
    np.array([  emptyCell, pawn, emptyCell, emptyCell, emptyCell, emptyCell, emptyCell, emptyCell]),
    np.array([  king, emptyCell, emptyCell, emptyCell, emptyCell, emptyCell, emptyCell, emptyCell]),
])
