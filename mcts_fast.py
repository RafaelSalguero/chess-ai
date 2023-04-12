import numpy as np
import math
from eval import evalBoard, evalWin
from layers import calc_layers, set_layer_input_data
from moves import allocate_moves_array, apply_move_inplace, flip_board, get_all_moves, move_str, undo_move_inplace
from ttable import get_transposition_table, set_transposition_table
from utils import fstr, get_np_hash, onehot_encode_board, onehot_encode_board_n
from numba import njit

from view import print_board

key_parent_index = 0
key_child_index = 1
key_child_count = 2
key_depth = 3
key_illegal_childs = 4

key_t = 0
key_n = 1
key_own_reward = 2

illegal_n = -1.0

win_bonus_ratio = 1.05

rollout_on_expand = False

@njit
def allocate(count):
    # (t, n, own_reward)
    values = np.zeros((count, 3), np.float32)

    # (parent_index, child_index, child_count, depth, illegal_childs)
    indices = np.zeros((count, 5), np.int32)

    moves = allocate_moves_array(count)
    undo_move = np.zeros((count, 2), np.int32)

    return (values, indices, moves, undo_move)

@njit
def is_expanded(indices, node_index):
    return indices[node_index, key_child_index] != 0

@njit
def get_color(indices, node_index):
    depth = indices[node_index, key_depth]
    return get_color_from_depth(depth)

@njit
def get_color_from_depth(depth):
    color = 1 if depth % 2 == 0 else -1
    return color

@njit
def has_parent(node_index):
    return node_index > 0

@njit
def has_childs(indices, node_index):
    return indices[node_index, key_child_count] > 0

@njit
def get_parent(indices, node_index):
    if(not has_parent(node_index)):
        return None
    return indices[node_index, key_parent_index]

@njit
def fill_node(values, indices, moves, undo_moves, parent_index, node_index, board, model, repetition_ttable, eval_ttable):
    undo_move = apply_move_inplace(board, moves[node_index])
    
    undo_moves[node_index, 0] = undo_move[0]
    undo_moves[node_index, 1] = undo_move[1]
    
    indices[node_index, key_parent_index] = parent_index
    depth = indices[parent_index, key_depth] + 1
    indices[node_index, key_depth] = depth

    node_color = get_color_from_depth(depth)
    (is_draw, _) = get_transposition_table(repetition_ttable, board, node_color, 1)
    if(is_draw):
        # A draw node has no childs, so we initialized as already expanded with 0 childs:
        indices[node_index, key_child_index] = node_index + 1
        indices[node_index, key_child_count] = 0

    if(rollout_on_expand):
        own_reward = 0.0 if is_draw else rollout(board, 1, node_color, None, eval_ttable)    
        values[node_index, key_own_reward] = own_reward

    is_win = evalWin(board) != 0

    undo_move_inplace(board, moves[node_index], undo_move)

    return is_win
    
@njit
def remove_illegal(values, indices, moves, node_index):
    """
        Remove the current node from the stats, the node will still be on the child list
    """

    backprop(values, indices, node_index, -values[node_index, key_n], -values[node_index, key_t])
    # Remove all childs, since an illegal move can't have following moves:
    # indices[node_index, key_child_count] = 0
    values[node_index, key_n] = illegal_n
    
    parent_index = indices[node_index, key_parent_index]
    indices[parent_index, key_illegal_childs] += 1

    if(indices[parent_index, key_child_count] == indices[parent_index, key_illegal_childs]):
        # No more moves in parent, so this is a checkmate
        win_reward = get_color(indices, node_index)
        values[parent_index, key_own_reward] = win_reward
        # print(f"checkmate found in {pv_str(indices, moves, parent_index)}")    

@njit
def expand(values, indices, moves, undo_moves, node_index, first_child_index, board, model, repetition_ttable, eval_ttable):
    if(first_child_index > (len(moves) - 128)):
        print(f"first child index {first_child_index} limit is close to {len(moves)} !")
    last_move_index = get_all_moves(board, get_color(indices, node_index), moves, first_child_index)

    child_count = last_move_index - first_child_index 

    indices[node_index, key_child_index] = first_child_index
    indices[node_index, key_child_count] = child_count

    is_check = False
    for child_index in range(first_child_index, last_move_index):
        is_win = fill_node(values, indices, moves, undo_moves, node_index, child_index, board, model, repetition_ttable, eval_ttable)
        is_check = is_check or is_win
        
    return not is_check

@njit
def undo_moves_rec(indices, moves, undo_moves, node_index, board):
    while has_parent(node_index):
        undo_move_inplace(board, moves[node_index], undo_moves[node_index])
        node_index = indices[node_index, key_parent_index]

@njit
def explore(values, indices, moves, undo_moves, model, c: float, prior_weight, board, next_child_index, repetition_ttable, eval_ttable):
    current_index = 0

    while indices[current_index, key_child_index] != 0 and indices[current_index, key_child_count] > 0:
        current_index = pick_child_best_ucb(values, indices, moves, current_index, c, prior_weight)

        move = moves[current_index]
        apply_move_inplace(board, move)

    

    if values[current_index, key_n] > 0 and not is_expanded(indices, current_index):
        is_legal_move = expand(values, indices, moves, undo_moves, current_index, next_child_index, board, model, repetition_ttable, eval_ttable)

        child_count = indices[current_index, key_child_count]
        next_child_index += child_count

        if(not is_legal_move):
            undo_moves_rec(indices, moves, undo_moves, current_index, board)
            remove_illegal(values, indices, moves, current_index)
            return next_child_index

        if child_count > 0:
            current_index = indices[current_index, key_child_index]
            apply_move_inplace(board, moves[current_index])
        
    reward = 0
    if(rollout_on_expand):
        reward = values[current_index, key_own_reward]
    else:
        reward = rollout(board, 1, get_color(indices, current_index), model, eval_ttable)
    values[current_index, key_own_reward] = reward

    undo_moves_rec(indices, moves, undo_moves, current_index, board)

    backprop(values, indices, current_index, 1, reward)

    return next_child_index
    
@njit
def backprop(values, indices, node_index, n, t):
    while True:
        values[node_index, key_n] += n
        values[node_index, key_t] += t

        if(not has_parent(node_index)):
            break
        node_index = indices[node_index, key_parent_index]

@njit
def pick_child_best_ucb(values, indices, moves, node_index, c, prior_reward):
    first_child_index = indices[node_index, key_child_index]
    child_count = indices[node_index, key_child_count]

    best_node_index = first_child_index
    max_value = -math.inf

    for child_index in range(first_child_index, first_child_index + child_count):
        value = get_ucb_score(values, indices, moves, child_index, c, prior_reward)
        if(value > max_value):
            max_value = value
            best_node_index = child_index

    return best_node_index

@njit
def get_ucb_score(values, indices, moves, node_index, c, prior_reward):
    n = values[node_index, key_n]
    if(n == 0):
        return math.inf
    if(n == illegal_n):
        #print(f"node {pv_str(indices, moves, node_index)} is illegal")
        return -math.inf
    
    t = values[node_index, key_t]
    depth = indices[node_index, key_depth]
    own_reward = values[node_index, key_own_reward]

    parent_index = indices[node_index, key_parent_index]
    parent_n = values[parent_index, key_n]
    return get_ucb_score_formula(own_reward, n, t, depth, parent_n, c, prior_reward)

@njit
def get_ucb_score_formula(own_reward, n, t, depth, parent_n, c, prior_reward):
    if(n == 0):
        return math.inf
    if(n == illegal_n):
        return -math.inf
    
    prior = own_reward * prior_reward
    color = get_color_from_depth(depth)

    node_score = ((prior + t) / n) * -color
    exploration_score = math.sqrt(math.log(parent_n) / n)

    return node_score + (c) * exploration_score

@njit
def pick_child_best_n(values, indices, node_index):
    first_child_index = indices[node_index, key_child_index]
    child_count = indices[node_index, key_child_count]

    best_node_index = first_child_index
    max_value = -math.inf

    for child_index in range(first_child_index, first_child_index + child_count):
        value = values[child_index, key_n]
        if(value > max_value):
            max_value = value
            best_node_index = child_index

    return best_node_index

@njit
def eval_to_prob(x):
    """
        Maps from [-150, 150] to [-1, 1]
    """
    return np.cbrt(x) / np.cbrt(150)

@njit
def encode_ttable_value(x):
    return np.round(np.power(x * np.cbrt(900), 3))

@njit
def decode_ttable_value(x):
    return np.cbrt(x) / np.cbrt(900)

@njit
def model_eval(model, node_color, board):
    b = board
    
    if(node_color == -1):
        b = flip_board(board)
    
    nn_eval = True
    y = 0
    
    onehot_encode_board_n(b, model[0].data3d[0])
    y = calc_layers(model).data1d[0][0]
    if(nn_eval):

        y = (y - 0.5) * 2 * node_color

        #print(f"eval model: {fstr(y)}")
        #print(f"eval board: {fstr(eval_to_prob(evalBoard(board, 1)))}")
        #print_board(board)
    else:
        y = eval_to_prob(evalBoard(board, 1))

    return y

@njit
def rollout(board, eval_color, node_color, model, eval_ttable):
    win_val = evalWin(board) * win_bonus_ratio

    if(win_val == 0):
        (match, value) = get_transposition_table(eval_ttable, board, node_color, 1)

        if(match):
            #print(f"ttable match {fstr(decode_ttable_value(value))} {fstr(y)} ")
            return decode_ttable_value(value)

        y = model_eval(model, node_color, board)
        set_transposition_table(eval_ttable, board, node_color, 1, np.round(encode_ttable_value(y)))

        return y
    else:
        y = win_val * eval_color

    return y
    
@njit
def pv_str(indices, moves, node_index):
    parent_index = indices[node_index, key_parent_index]
    if(parent_index == node_index):
        return ""
    return pv_str(indices, moves, parent_index) + "->" + move_str(moves[node_index])

@njit
def add_repetition_table_entry(board, move, ttable):
    set_transposition_table(ttable, board, 1, 1, 1)

    undo_move = apply_move_inplace(board, move)
    
    set_transposition_table(ttable, board, -1, 1, 1)

    undo_move_inplace(board, move, undo_move)

@njit
def mcts(board, iterations, repetition_ttable, eval_ttable, model, c = 1, prior_weight = 2, verbose = False):
    next_child_index = 1
    max_size = iterations * 15
    (values, indices, moves, undo_moves) = allocate(max_size)
    set_layer_input_data(np.zeros((8,8,8), np.float32), model)

    for it in range(iterations):
        next_child_index = explore(values, indices, moves, undo_moves, model, c, prior_weight, board, next_child_index, repetition_ttable, eval_ttable)

    best_child_index = pick_child_best_n(values, indices, 0)

    bc = 0
    while True:
        if verbose and has_childs(indices, bc):
            first_child_index = indices[bc, key_child_index]
            child_count = indices[bc, key_child_count]
            for child_index in range(first_child_index, first_child_index + child_count):
                print(f"{pv_str(indices, moves, child_index)} (n: {fstr(values[child_index, key_n])}) (t: {fstr(values[child_index, key_t])}) (own_reward: {round(values[child_index, key_own_reward] * 100)}%, reward: {round((values[child_index, key_t] / (values[child_index, key_n] + 0.01)) * 100)}%) node_score: {fstr(get_ucb_score(values, indices, moves, child_index, 0, 0))}")

        if(has_childs(indices, bc)):
            bc = pick_child_best_n(values, indices, bc)
        else:
            print(f"pv: {pv_str(indices, moves, bc)}")
            break

    print(f"count: {next_child_index} / {max_size}")
    
    # Fill repetition table:
    best_move = moves[best_child_index]

    add_repetition_table_entry(board, best_move, repetition_ttable)
    return best_move
    
