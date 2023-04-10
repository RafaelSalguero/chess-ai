import numpy as np
import math
from eval import evalBoard, evalWin
from moves import allocate_moves_array, apply_move_inplace, flip_board, get_all_moves, move_str, undo_move_inplace
from utils import fstr

key_parent_index = 0
key_child_index = 1
key_child_count = 2
key_depth = 3

key_t = 0
key_n = 1
key_own_reward = 2

def allocate(count):
    # (t, n, own_reward)
    values = np.zeros((count, 3), np.float32)

    # (parent_index, child_index, child_count, depth)
    indices = np.zeros((count, 4), np.int32)

    moves = allocate_moves_array(count)
    undo_move = np.zeros((count, 2), np.int32)

    return (values, indices, moves, undo_move)

def is_expanded(indices, node_index):
    return indices[node_index, key_child_index] != 0

def get_color(indices, node_index):
    depth = indices[node_index, key_depth]
    color = 1 if depth % 2 == 0 else -1
    return color

def has_parent(node_index):
    return node_index > 0

def has_childs(indices, node_index):
    return indices[node_index, key_child_count] > 0

def get_parent(indices, node_index):
    if(not has_parent(node_index)):
        return None
    return indices[node_index, key_parent_index]

def fill_node(values, indices, moves, undo_moves, parent_index, node_index, board):
    undo_move = apply_move_inplace(board, moves[node_index])
    
    undo_moves[node_index, 0] = undo_move[0]
    undo_moves[node_index, 1] = undo_move[1]

    own_reward = rollout(board, 1, None)

    undo_move_inplace(board, moves[node_index], undo_move)

    values[node_index, key_own_reward] = own_reward
    
    indices[node_index, key_parent_index] = parent_index
    indices[node_index, key_depth] = indices[parent_index, key_depth] + 1

def expand(values, indices, moves, undo_moves, node_index, first_child_index, board):
    last_move_index = get_all_moves(board, get_color(indices, node_index), moves, first_child_index)

    child_count = last_move_index - first_child_index 

    indices[node_index, key_child_index] = first_child_index
    indices[node_index, key_child_count] = child_count

    for child_index in range(first_child_index, last_move_index):
        fill_node(values, indices, moves, undo_moves, node_index, child_index, board)

def undo_moves_rec(indices, moves, undo_moves, node_index, board):
    while has_parent(node_index):
        print(f"undo move {move_str(moves[node_index])}")
        undo_move_inplace(board, moves[node_index], undo_moves[node_index])
        node_index = indices[node_index, key_parent_index]

def explore(values, indices, moves, undo_moves, c: float, prior_weight, board, next_child_index):
    current_index = 0
    print("explore 1")

    while indices[current_index, key_child_index] != 0 and indices[current_index, key_child_count] > 0:
        current_index = pick_child_best_ucb(values, indices, current_index, c, prior_weight)

        move = moves[current_index]
        apply_move_inplace(board, move)

    print("explore 2")
    if values[current_index, key_n] > 0 and not is_expanded(indices, current_index):
        print("explore 2.1")

        print(f"next_child_index: {next_child_index}, size: {len(moves)}")
        expand(values, indices, moves, undo_moves, current_index, next_child_index, board)
        
        print("explore 2.2")

        undo_moves_rec(indices, moves, undo_moves, current_index, board)

        print("explore 2.3")


        child_count = indices[current_index, key_child_count]
        print(f"expand child count: {child_count}")
        next_child_index += child_count

        if child_count > 0:
            current_index = indices[current_index, key_child_index]
    else:
        print("explore 3")

        undo_moves_rec(indices, moves, undo_moves, current_index, board)
        print("explore 4")


    reward = values[current_index, key_own_reward]
    backprop(values, indices, current_index, 1, reward)

    return next_child_index
    
def backprop(values, indices, node_index, n, t):
    while True:
        values[node_index, key_n] += n
        values[node_index, key_t] += t

        if(not has_parent(node_index)):
            break
        node_index = indices[node_index, key_parent_index]

def pick_child_best_ucb(values, indices, node_index, c, prior_reward):
    first_child_index = indices[node_index, key_child_index]
    child_count = indices[node_index, key_child_count]

    best_node_index = first_child_index
    max_value = -math.inf

    for child_index in range(first_child_index, first_child_index + child_count):
        value = get_ucb_score(values, indices, child_index, c, prior_reward)
        if(value > max_value):
            max_value = value
            best_node_index = child_index

    return best_node_index

def get_ucb_score(values, indices, node_index, c, prior_reward):
    n = values[node_index, key_n]
    if(n == 0):
        return math.inf
    
    prior = values[node_index, key_own_reward] * prior_reward
    t = values[node_index, key_t]
    depth = indices[node_index, key_depth]
    color = get_color(indices, node_index)

    parent_index = indices[node_index, key_parent_index]
    parent_n = values[parent_index, key_n]

    node_score = ((prior + t) / n) * -color
    exploration_score = math.sqrt(math.log(parent_n) / n)

    return node_score + (c / depth) * exploration_score

def pick_child_best_n(values, indices, node_index):
    first_child_index = indices[node_index, key_child_index]
    child_count = indices[node_index, key_child_count]

    if(child_count == 0):
        return None

    best_node_index = first_child_index
    max_value = -math.inf

    for child_index in range(first_child_index, first_child_index + child_count):
        value = values[child_index, key_n]
        if(value > max_value):
            max_value = value
            best_node_index = child_index

    return best_node_index
def eval_to_prob(x):
    """
        Maps from [-150, 150] to [-1, 1]
    """
    return np.cbrt(x) / np.cbrt(150)

def model_eval(model, color, board):
    b = board
    
    if(color == -1):
        b = flip_board(board)
    
    nn_eval = False
    y = 0
    if(nn_eval):
        # y = internal_model_eval(model, onehot_encode_board(b).reshape((-1, 8, 8, 8))).numpy().reshape(-1)[0]
        # y = (y - 0.5) * 2
        y = 0.0
    else:
        y = eval_to_prob(evalBoard(board, 1))

    return y

def rollout(board, color, model):
    win_val = evalWin(board) * 1000

    if(win_val == 0):
        y = model_eval(model, color, board)
    else:
        y = win_val * color

    return y
    
def pv_str(indices, moves, node_index):
    parent_index = indices[node_index, key_parent_index]
    if(parent_index == node_index):
        return ""
    return pv_str(indices, moves, parent_index) + "->" + move_str(moves[node_index])

def mcts(board, iterations, c = 1, prior_weight = 2, verbose = False):
    next_child_index = 1
    max_size = iterations * 3
    (values, indices, moves, undo_moves) = allocate(max_size)

    for it in range(iterations):
        print(f"iteration {it}")
        next_child_index = explore(values, indices, moves, undo_moves, c, prior_weight, board, next_child_index)

    best_child_index = pick_child_best_n(values, indices, 0)

    bc = 0
    while bc is not None:
        if verbose and has_childs(indices, bc):
            first_child_index = indices[bc, key_child_index]
            child_count = indices[bc, key_child_count]
            for child_index in range(first_child_index, first_child_index + child_count):
                print(f"{pv_str(indices, moves, child_index)} (n: {values[child_index, key_n]}) (t: {fstr(values[child_index, key_t])}) (own_reward: {round(values[child_index, key_own_reward] * 100)}%) node_score: {fstr(get_ucb_score(values, indices, child_index, 0, 0))}")

        next_bc = pick_child_best_n(values, indices, bc)
        if(next_bc is None):
            print(f"pv: {pv_str(indices, moves, bc)}")

        bc = next_bc

    print(f"count: {next_child_index} / {max_size}")
    print(f"best move: {best_child_index} ({moves[best_child_index]}) {indices[0, key_child_count]}")
    return moves[best_child_index]
    
