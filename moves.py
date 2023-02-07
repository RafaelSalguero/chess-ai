import numpy as np
from board import pawn, emptyCell
# A move is represented as  [[y_start, x_start], [y_end, x_end]]
# All moves are calculated for white, the board is flipped when calculating black moves

# Returns true if the position is inside the board, false otherwhite
col_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

# Convert a pos to string
def pos_str(pos):
    return col_names[pos[1]] + str(8 - pos[0])

def flip_pos(pos):
    return np.array([7 - pos[0], pos[1]])

def flip_move(move):
    return np.array(list(map(flip_pos, move)))

# Convert string to position
def str_pos(str):
    return np.array([8 - int(str[1]), col_names.index(str[0])])

def move_str(move):
    return pos_str(move[0]) + pos_str(move[1])

def str_move(str):
    return np.array([str_pos(str[0:2]), str_pos(str[2:4])])

def pos_inside(pos):
    return np.all(np.logical_and(pos >= np.array([0, 0]),  pos < np.array([8, 8])))

def flip_board(board):
    return -np.flipud(board)

def cell(board, pos):
    return board[pos[0], pos[1]]

def set_cell(board, pos, value):
    board[pos[0], pos[1]] = value

def is_empty_cell(board, pos):
    if(not pos_inside(pos)):
        return True
    
    return np.sum(cell(board, pos)) == 0

def has_piece(board, pos, white):
    if(is_empty_cell(board, pos)):
        return False
    
    is_white = np.sum(cell(board, pos)) > 0
    return is_white == white

# returns a new board after the move was applied
def apply_move(board, move):
    ret = np.copy(board)
    set_cell(ret, move[0], emptyCell)
    set_cell(ret, move[1], cell(board, move[0]))

    return ret;

def get_pawn_moves(board, pos):
    forward = [-1, 0]

    ret = []
    # move 1 forward:
    next_pos = pos + forward
    if(pos_inside(next_pos) and is_empty_cell(board, next_pos)):
        ret.append([pos, next_pos])
    
    # move 2 forward:
    start_y = 6
    next_pos = pos + forward + forward
    if(pos[0] == start_y and is_empty_cell(board, next_pos)):
        ret.append([pos, next_pos])

    # take:
    next_pos = pos + forward + np.array([0, 1])
    if(has_piece(board, next_pos, False)):
        ret.append([pos, next_pos])

    next_pos = pos + forward + np.array([0, -1])
    if(has_piece(board, next_pos, False)):
        ret.append([pos, next_pos])

    return ret

def get_all_moves(board):
    ret = []
    for x in range(0, 8):
        for y in range(0, 8):
            pos = np.array([x, y])
            cell = board[x, y]
            if(np.array_equal(cell, pawn)):
                ret += get_pawn_moves(board, pos)
    
    return ret