import numpy as np
from termcolor import colored
from colorama import Fore, Back
from board import rook, knight, bishop, queen, king, pawn, emptyCell, piecesRank, emptyRank, initialBoard

def print_piece(piece, on_white):
    black = np.sum(piece) < 0
    piece = np.abs(piece)

    all_pieces = np.array([rook, knight, bishop, queen, king, pawn, emptyCell])

    pieceIndex = np.where(np.all(all_pieces ==  piece, axis=1))[0][0]
    black_chars = ['♜', '♞', '♝', '♛', '♚', '♟︎', ' ']
    white_chars = ['♖', '♘', '♗', '♕', '♔', '♙', ' ']

    char =  (black_chars if black else white_chars)[pieceIndex]
    print(' ' + char + ' ', end='')

def print_rank(rank, white):
    color = white
    for piece in rank:
        print(Back.WHITE if color else Back.GREEN, end='')
        print_piece(piece, color)
        color = not color
    print(Back.RESET)

def print_board(board):
    color = True
    rn = 8
    for rank in board:
        print(rn, end='')
        rn = rn - 1
        print_rank(rank, color)
        color = not color
    
    print(' ', end='')
    for col in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
        print(' ' + col + ' ', end='')
    print()

if __name__ == '__main__':
    print_board(initialBoard)