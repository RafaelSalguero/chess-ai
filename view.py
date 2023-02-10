import numpy as np
from colorama import Fore, Back
from board import rook, knight, bishop, queen, king, pawn, emptyCell
from moves import col_names

black_chars = ['♜', '♞', '♝', '♛', '♚', '♟︎', ' ']
white_chars = ['♖', '♘', '♗', '♕', '♔', '♙', ' ']
all_pieces = np.array([rook, knight, bishop, queen, king, pawn, emptyCell])

def print_piece(piece, on_white):
    black = np.sum(piece) < 0
    piece = np.abs(piece)

    pieceIndex = np.where(np.all(all_pieces ==  piece, axis=1))[0][0]

    char =  (black_chars if black else white_chars)[pieceIndex]
    print(' ' + char + ' ', end='')

def parse_piece(piece):
    pchar = piece.strip(" ")

    if(pchar == ''):
        return emptyCell
    
    if(pchar in black_chars):
        return -all_pieces[black_chars.index(pchar)]
    
    if(pchar in white_chars):
        return all_pieces[white_chars.index(pchar)]
    
    raise Exception(f"Cant parse piece '{piece}'")

def print_rank(rank, white):
    color = white
    for piece in rank:
        print(Back.WHITE if color else Back.GREEN, end='')
        print_piece(piece, color)
        color = not color
    print(Back.RESET)

def parse_rank(rank):
    rank = rank.lstrip(" ")
    ret = []
    for col in range(0, 8):
        piece = rank[(col * 3 + 1):(col * 3 + 1 + 3)]
        ret.append(parse_piece(piece))
    
    return np.array(ret)

def print_board(board):
    color = True
    rn = 8
    for rank in board:
        print(rn, end='')
        rn = rn - 1
        print_rank(rank, color)
        color = not color
    
    print(' ', end='')
    for col in col_names:
        print(' ' + col + ' ', end='')
    print()

def parse_board(board):
    lines = board.splitlines(False)
    ret = []
    for line in lines:
        if(len(line) < 3 * 8 + 1):
            continue
        if(not line[0:1].isnumeric()):
            continue

        ret.append(parse_rank(line))

    return np.array(ret)