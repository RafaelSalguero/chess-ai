import numpy as np
from termcolor import colored
from colorama import Fore, Back
from board import rook, knight, bishop, queen, king, pawn, emptyCell, piecesRank, emptyRank, initialBoard
from view import print_board

if __name__ == '__main__':
    t = np.copy(initialBoard)
    t[0][1] = emptyCell
    print_board(np.flipud(t))