import numpy as np
from board import pawn, emptyCell, king, rook, knight, bishop, queen
from eval import evalWin
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

def has_piece(board, pos, color):
    if(is_empty_cell(board, pos)):
        return False
    
    return np.sign(np.sum(cell(board, pos))) == color

# returns a new board after the move was applied
def apply_move(board, move):
    ret = np.copy(board)
    apply_move_inplace(ret, move)
    return ret

# apply a move in the given board, returns an object that can be passed to undo_move_inplace
def apply_move_inplace(board, move):
    orig_cells = np.copy([cell(board, move[0]), cell(board, move[1])])

    end_piece = orig_cells[0]
    
    # promotions:
    if(move[1][0] == 0 and np.array_equal(end_piece, pawn)):
        end_piece = queen

    if(move[1][0] == 7 and np.array_equal(end_piece, -pawn)):
        end_piece = -queen


    set_cell(board, move[0], emptyCell)
    set_cell(board, move[1], end_piece)
    
    return orig_cells

def undo_move_inplace(board, move, undo):
    set_cell(board, move[0], undo[0])
    set_cell(board, move[1], undo[1])

def get_pawn_moves(board, pos, color):
    forward = np.array([-1, 0]) * color

    ret = []
    # move 1 forward:
    next_pos = pos + forward
    if(pos_inside(next_pos) and is_empty_cell(board, next_pos)):
        ret.append(next_pos)
    
    # move 2 forward:
    start_y = 6 if color==1 else 1
    next_pos = pos + forward + forward
    if(pos[0] == start_y and is_empty_cell(board, next_pos)):
        if(not pos_inside(next_pos)):
            raise Exception("pos")
        ret.append(next_pos)

    # take:
    next_pos = pos + forward + np.array([0, 1])
    if(has_piece(board, next_pos, -color)):
        ret.append(next_pos)

    next_pos = pos + forward + np.array([0, -1])
    if(has_piece(board, next_pos, -color)):
        ret.append(next_pos)

    return ret

bishop_vectors = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
rook_vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
queen_vectors = np.concatenate((bishop_vectors, rook_vectors))

def get_king_moves(board, pos, color):
    ret = []
    all_next_pos = queen_vectors + pos
    for next_pos in all_next_pos:
        if(not pos_inside(next_pos)):
            continue
        if(has_piece(board, next_pos, color)):
            continue
        ret.append(next_pos)

    return ret

def get_ray_moves(board, pos, forward, color):
    ret = []
    while True:
        pos = pos + forward
        # check bounds or piece blocking
        if((not pos_inside(pos)) or has_piece(board, pos, color)):
            break
        
        ret.append(pos)

        # take
        if(has_piece(board, pos, -color)):
            break
    return ret

def get_vector_moves(board, pos, vectors, color):
    ret = []
    for vector in vectors:
        ret += get_ray_moves(board, pos, vector, color)
    return ret

def get_all_moves(board, color):
    
    ret = []

    if(evalWin(board) != 0): 
        return ret
    
    for y in range(0, 8):
        for x in range(0, 8):
            pos = np.array([y, x])
            cell = board[y, x] * color
            curr_ret = []
            if(np.array_equal(cell, pawn)):
                curr_ret += get_pawn_moves(board, pos, color)
            elif (np.array_equal(cell, king)):
                curr_ret += get_king_moves(board, pos, color)
            elif (np.array_equal(cell, queen)):
                curr_ret += get_vector_moves(board, pos, queen_vectors, color)

            ret += list(map(lambda np: [pos, np], curr_ret))
    
    return ret