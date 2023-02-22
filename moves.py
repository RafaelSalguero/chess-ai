import numpy as np
from board import pawn, emptyCell, king, rook, knight, bishop, queen, rook_moved, king_moved
from eval import evalWin
from numba import njit
import numpy.typing as npt
import tensorflow as tf

# A move is represented as  [[y_start, x_start], [y_end, x_end]]
# All moves are calculated for white, the board is flipped when calculating black moves

# Returns true if the position is inside the board, false otherwhite
col_names = 'abcdefgh'

@njit
def rank_str(rank):
    return str(8 - rank)

@njit
def file_str(file): 
    return col_names[file]

# Convert a pos to string
@njit
def pos_str(pos):
    return file_str(pos[1]) + rank_str(pos[0])

@njit
def flip_pos(pos):
    return np.array([7 - pos[0], pos[1]])

@njit
def flip_move(move):
    return np.array(list(map(flip_pos, move)))

# Convert string to position
@njit
def str_pos(str):
    return np.array([8 - int(str[1]), col_names.index(str[0])])

@njit
def move_str(move):
    return pos_str(move[0]) + pos_str(move[1])

@njit
def is_piece(cell, piece, color = None, strict = False):
    if(not strict and piece == king and is_piece(cell, king_moved, color, True)):
        return True 
    
    if(not strict and piece == rook and is_piece(cell, rook_moved, color, True)):
        return True 
        
    c = cell

    if(color == None):
        c = np.abs(c)
    else:
        c = c * color
    
    return c == piece

# Converts a move to algebraic notation
@njit
def move_str_an(board, all_moves, move):
    orig = cell(board, move[0])
    color = np.abs(orig)

    if(is_piece(orig, king, color, True)):
        if(move[1][1] == 2):
            return "O-O-O"
        if(move[1][1] == 4):
            return "O-O"
        
    name =  "K" if is_piece(orig, king) else "Q" if is_piece(orig, queen) else  "R" if is_piece(orig, rook) else "N" if is_piece(orig, knight) else "B" if is_piece(orig, bishop) else ""
    take = not is_empty_cell(board, move[1])
    take_str = 'x' if take else ''
    dest_str = pos_str(move[1])

    if(name == ""):
        if(take): return col_names[move[0][1]] + take_str + dest_str
        return dest_str
    
    # If two pices from the same type can move to the same dest:
    conflict = False
    conflict_same_file = False
    for m in all_moves:
        if(np.array_equal(m[0], move[0]) and np.array_equal(m[1], move[1])):
            continue

        if(np.array_equal(cell(board, m[0]), orig) and np.array_equal(m[1], move[1])):
            conflict = True

            if(m[0][1] == move[0][1]):
                conflict_same_file = True
            
    # If there are any conflicts in the same file:
    if(conflict_same_file):
        return name + rank_str(move[0][0]) + take_str + dest_str
    elif (conflict):
        return name + file_str(move[0][0]) + take_str + dest_str
    
    return name + take_str + dest_str

@njit
def moves_str_an(board, moves):
    ret = []
    for move in moves:
        ret.append(move_str_an(board, moves, move))

    return ret

def str_move_an(board, all_moves, str):
    for move in all_moves:
        if(move_str_an(board, all_moves, move) == str):
            return move
        
    raise Exception("Move was not found")

@njit
def str_move(str):
    return np.array([str_pos(str[0:2]), str_pos(str[2:4])])

@njit
def pos_inside(pos):
    return np.all(np.logical_and(pos >= np.array([0, 0]),  pos < np.array([8, 8])))

@njit
def flip_board(board):
    return -np.flipud(board)

@njit
def cell(board, pos):
    return board[pos[0], pos[1]]

@njit
def set_cell(board, pos, value):
    board[pos[0], pos[1]] = value

@njit
def is_empty_cell(board, pos):
    if(not pos_inside(pos)):
        return True
    
    return cell(board, pos) == 0

@njit
def has_piece(board, pos, color):
    if(is_empty_cell(board, pos)):
        return False
    
    return np.sign(cell(board, pos)) == color

# returns a new board after the move was applied
@njit
def apply_move(board, move):
    ret = np.copy(board)
    apply_move_inplace(ret, move)
    return ret

# apply a move in the given board, returns an object that can be passed to undo_move_inplace
@njit
def apply_move_inplace(board, move):
    orig_cells = (cell(board, move[0]), cell(board, move[1]))
    color = np.sign(orig_cells[0])

    orig_piece = orig_cells[0] 
    end_piece = orig_piece

    if(np.abs(orig_piece) == rook):
        end_piece = rook_moved * color

    if(np.abs(orig_piece) == king):
        end_piece = king_moved * color
    
    # promotions:
    if(move[1][0] == 0 and np.array_equal(end_piece, pawn)):
        end_piece = queen

    if(move[1][0] == 7 and np.array_equal(end_piece, -pawn)):
        end_piece = -queen

    # mark as moved:
    set_cell(board, move[0], emptyCell)
    set_cell(board, move[1], end_piece)

    if(np.abs(orig_piece) == king):
        if(move[1][1] == 2):
            # left castle
            set_cell(board, np.array([move[0][0], 0]), emptyCell)
            set_cell(board, np.array([move[0][0], 3]), rook_moved * color)

        if(move[1][1] == 6):
            # right castle
            set_cell(board, np.array([move[0][0], 7]), emptyCell)
            set_cell(board, np.array([move[0][0], 5]), rook_moved * color)
    
    return orig_cells

@njit
def undo_move_inplace(board, move, undo):
    orig_piece = undo[0] 
    color = np.sign(orig_piece)

    set_cell(board, move[0], undo[0])
    set_cell(board, move[1], undo[1])

    if(np.abs(orig_piece) == king):
        if(move[1][1] == 2):
            # left castle
            set_cell(board, np.array([move[0][0], 0]), rook * color)
            set_cell(board, np.array([move[0][0], 3]), emptyCell)

        if(move[1][1] == 6):
            # right castle
            set_cell(board, np.array([move[0][0], 7]), rook * color)
            set_cell(board, np.array([move[0][0], 5]), emptyCell)

@njit
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
    if(pos[0] == start_y and is_empty_cell(board, pos + forward) and is_empty_cell(board, next_pos)):
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

@njit
def get_king_moves(board, pos, color):
    ret = []
    for vector in queen_vectors:
        next_pos = pos + vector
        if(not pos_inside(next_pos)):
            continue
        if(has_piece(board, next_pos, color)):
            continue
        ret.append(next_pos)

    # Castling:
    l_rook_pos = np.array([pos[0], 0])
    l_rook_pos_e1 = np.array([pos[0], 1])
    l_rook_pos_e2 = np.array([pos[0], 2])
    l_rook_pos_e3 = np.array([pos[0], 3])


    if(is_piece(cell(board, pos), king, color, True) and 
       is_piece(cell(board, l_rook_pos), rook, color, True) and 
       is_empty_cell(board, l_rook_pos_e1) and 
       is_empty_cell(board, l_rook_pos_e2)  and 
       is_empty_cell(board, l_rook_pos_e3)
    ):
        ret.append(l_rook_pos_e2)

    r_rook_pos = np.array([pos[0], 7])
    r_rook_pos_e1 = np.array([pos[0], 5])
    r_rook_pos_e2 = np.array([pos[0], 6])

    if(is_piece(cell(board, pos), king, color, True) and 
       is_piece(cell(board, r_rook_pos), rook, color, True) and 
       is_empty_cell(board, r_rook_pos_e1) and 
       is_empty_cell(board, r_rook_pos_e2)
    ):
        ret.append(r_rook_pos_e2)

    return ret

@njit
def get_ray_moves(board: npt.NDArray[np.float32], pos: npt.NDArray[np.int64], forward: npt.NDArray[np.int64], color: np.float32) -> list[npt.NDArray[np.int64]]:
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

@njit
def get_vector_moves(board, pos, vectors, color) -> list[npt.NDArray[np.int64]]:
    ret = [np.array([0,0], dtype=np.int64) for x in range(0)]
    for vector in vectors:
        ret += get_ray_moves(board, pos, vector, color)
    return ret

@njit
def get_knight_moves(board, pos, color):
    vectors = np.array([
        [2, 1],
        [1, 2],

        [-2, 1],
        [-1, 2],

        [-2, -1],
        [-1, -2],

        [2, -1],
        [1, -2],
        ])

    ret = []
    for vector in vectors:
        next_pos = pos + vector
        if(not pos_inside(next_pos) or has_piece(board, next_pos, color)):
            continue
        
        ret.append(next_pos)

    return ret

# Get a list of all possible moves, this is a pure function since moves are indexable
@njit
def get_all_moves(board, color):
    ret = [(np.array([0,0]), np.array([0,0])) for x in range(0)]

    if(evalWin(board) != 0): 
        return ret
    
    for y in range(0, 8):
        for x in range(0, 8):
            pos = np.array([y, x])
            cell = board[y, x]
            curr_ret = [np.array([0,0]) for x in range(0)]
            if(is_piece(cell, pawn, color)):
                curr_ret += get_pawn_moves(board, pos, color)
            elif (is_piece(cell, knight, color)):
                curr_ret += get_knight_moves(board, pos, color)
            elif (is_piece(cell, king, color)):
                curr_ret += get_king_moves(board, pos, color)
            elif (is_piece(cell, queen, color)):
                curr_ret += get_vector_moves(board, pos, queen_vectors, color)
            elif (is_piece(cell, bishop, color)):
                curr_ret += get_vector_moves(board, pos, bishop_vectors, color)
            elif (is_piece(cell, rook, color)):
                curr_ret += get_vector_moves(board, pos, rook_vectors, color)

            for next_pos in curr_ret:
                ret.append((pos, next_pos))
    
    return ret