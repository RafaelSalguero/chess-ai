import numpy as np
from colorama import Fore, Back
from board import rook, knight, bishop, queen, king, pawn, emptyCell

def print_piece(piece):
    black = np.sum(piece) < 0
    piece = np.abs(piece)

    all_pieces = np.array([rook, knight, bishop, queen, king, pawn, emptyCell])

    pieceIndex = np.where(np.all(all_pieces ==  piece, axis=1))[0][0]
    chars = ['♜', '♞', '♝', '♛', '♚', '♟︎', ' ']

    char = chars[pieceIndex]
    print(Fore.BLACK if black else Fore.WHITE + char, end='')

def print_rank(rank, white):
    color = white
    for piece in rank:
        print(Back.CYAN if color else Back.BLUE + '   ', end='')
        color = not color
    print(Back.RESET + '')

    color = white
    for piece in rank:
        print(Back.CYAN if color else Back.BLUE + ' ', end='')
        print_piece(piece)
        print(' ', end='')
        color = not color
    print(Back.RESET + '')

    color = white
    for piece in rank:
        print(Back.CYAN if color else Back.BLUE + '   ', end='')
        color = not color
    print(Back.RESET + '')
    
    
