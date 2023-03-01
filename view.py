import numpy as np
from board import rook, knight, bishop, queen, king, pawn, emptyCell, rook_moved, king_moved
from numba import njit


CSI = '\033['

@njit
def code_to_chars(code):
    return CSI + str(code) + 'm'

BACK_GREEN = code_to_chars(42)
BACK_WHITE = code_to_chars(47)
BACK_RESET = code_to_chars(49)

ascii_chars = 'rrnbqkkp '
black_chars = '♜♜♞♝♛♚♚♟ ' 
white_chars = '♖♖♘♗♕♔♔♙ '

def board_to_ascii(s):
    for i, x in enumerate(black_chars):
        s = s.replace(x, ascii_chars[i].upper())

    for i, x in enumerate(white_chars):
        s = s.replace(x, ascii_chars[i])
    return s


@njit
def print_piece(piece, on_white):
    all_pieces = np.array([rook, rook_moved, knight, bishop, queen, king, king_moved, pawn, emptyCell])

    black = piece < 0
    piece = np.abs(piece)

    pieceIndex = np.where(all_pieces == piece)[0][0]

    char =  (black_chars if black else white_chars)[pieceIndex]
    return ' ' + char + ' '

def parse_piece(piece):
    all_pieces = np.array([rook_moved, rook_moved, knight, bishop, queen, king_moved, king_moved, pawn, emptyCell])

    pchar = piece.strip(" ")

    if(pchar == ''):
        return emptyCell
    
    color = 1 if pchar.lower() == pchar else -1
    
    pchar = pchar.lower()
    if(pchar in ascii_chars):
        return all_pieces[ascii_chars.index(pchar)] * color
    
    raise Exception(f"Cant parse piece '{piece}'")

@njit
def print_rank(rank, white):
    color = white
    ret = ''
    for piece in rank:
        ret += (BACK_WHITE if color else BACK_GREEN)
        ret += print_piece(piece, color)
        color = not color
    ret += (BACK_RESET)
    return ret

def parse_rank(rank):
    rank = rank.lstrip(" ")
    ret = []

    for col in range(0, 8):
        piece = rank[(col * 3 + 1):(col * 3 + 1 + 3)]
        ret.append(parse_piece(piece))
    
    return np.array(ret)

@njit
def print_board(board):
    color = True
    rn = 8
    for rank in board:
        rn = rn - 1
        print(f'{rn + 1}{print_rank(rank, color)}')
        color = not color
    
    print('  a  b  c  d  e  f  g  h')

def  parse_board(board):
    board = board_to_ascii(board)
    lines = board.splitlines(False)
    ret = []
    for line in lines:
        if(len(line) < 3 * 8 + 1):
            continue
        if(not line[0:1].isnumeric()):
            continue

        ret.append(parse_rank(line))

    return np.array(ret)