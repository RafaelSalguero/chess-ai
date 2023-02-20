import tensorflow as tf
import numpy as np
import itertools as it


emptyCell = np.int32(0)

rook = np.int32(1)
rook_moved = np.int32(2)

knight = np.int32(3)
bishop = np.int32(4)
queen = np.int32(5)

king = np.int32(6)
king_moved = np.int32(7)

pawn = np.int32(8)


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