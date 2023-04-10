from collections import OrderedDict
import numpy as np
import math
from eval import evalBoard, evalWin
from moves import allocate_moves_array, apply_move, apply_move_inplace, flip_board, get_all_moves, get_all_moves_slow, move_str, undo_move_inplace
from utils import fstr, onehot_encode_board
from view import print_board
import tensorflow as tf

class Node:
    def __init__(self, parent, childs, own_reward, is_checkmate, color, move, undo_move, depth):
        self.parent = parent
        self.childs = childs
        self.move = move
        self.undo_move = undo_move
        self.color = color
        self.depth = depth
        
        # total win count
        self.t = 0
        # visit count
        self.n = 0
        self.prior = 0
        self.own_reward = own_reward
        self.is_checkmate = is_checkmate
        self.is_check = False
        self.childs_check_count = 0

def expand(_self, color,  model, prior_weight, board):
    if(_self.childs is not None):
        print("This node is already expanded")
    
    _self.childs = get_childs(_self, color, model, prior_weight, board)

def remove_check(_self):
    """
        Remove the current node from the stats, the node will still be on the child list
    """
    if(_self.parent is None):
        return
    print(f"removing check {pv_str(_self)}")
    backprop(_self, -_self.n, -_self.t)
    
    _self.parent.childs_check_count += 1

    if(len(_self.parent.childs) == _self.parent.childs_check_count):
        # No more moves in parent, so this is a checkmate
        _self.parent.own_reward = 1.0
        print(f"checkmate found in {pv_str(_self.parent)}")

def explore(_self, c: float, model, prior_weight, board):
    current = _self
    # Search for the best child until we find a leaf node
    while current.childs is not None and len(current.childs) > 0:
        current = pick_child_best_ucb(current.childs, c)
        
        move = current.move
        apply_move_inplace(board, move)


    if not current.is_check and current.n > 0 and current.childs is None:
        expand(current, _self.color, model, prior_weight, board)
        undo_moves(current, board)
        
        if(current.is_check):
            remove_check(current)
            return
        
        if len(current.childs) > 0:
            current = current.childs[0]
    else:
        undo_moves(current, board)

    if(current.is_check):
        return
    
    reward = current.own_reward #rollout(current, self.color, model)
    current.own_reward = reward
    backprop(current, 1, reward)

    
def undo_moves(node: Node, board):
    while node is not None and node.parent is not None:
        undo_move_inplace(board, node.move, node.undo_move)
        node = node.parent

def backprop(node: Node, n, t):
    parent = node
    while parent is not None:
        parent.n = parent.n + n
        parent.t += t
        parent = parent.parent
    
def create_node(board, move, parent, prior_weight, eval_color, model):
    undo_move = apply_move_inplace(board, move)

    own_reward = rollout(board, eval_color, model)
    is_checkmate = evalWin(board) != 0

    undo_move_inplace(board, move, undo_move)

    ret = Node(parent, None, own_reward, is_checkmate, -parent.color, move, undo_move, parent.depth + 1)
    #ret = Node(None, None, 0.0, False, 1, move, undo_move, 1)

    # prior probability:
    ret.prior = ret.own_reward * prior_weight
    if ret.is_checkmate:
        parent.is_check = True

    return ret
    
def get_childs(parent: Node, eval_color, model, prior_weight, board):
    """
        Gets node children and sets the is_check flag for the given node
    """
    moves = get_all_moves_slow(board, parent.color)
    
    childs = []
    for move in moves:
        childs.append(create_node(board, move, parent, prior_weight, eval_color, model))

    return childs

@tf.function    
def internal_model_eval(model, x):
    return model(x)

def from_model_space(x):
    return np.power((x - 0.5) * (2 * np.cbrt(150)), 3)

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
    win_val = evalWin(board) * 1.05

    if(win_val == 0):
        y = model_eval(model, color, board)
    else:
        y = win_val * color

    return y

def get_ucb_score(node: Node, c: float):
    """
        Gets the UCB score, the tree search will explore nodes with higher scores first
    """

    parent = node.parent

    if(node.is_check):
        return -math.inf
    
    if (node.n == 0):
        return math.inf
    
    node_score = ((node.prior + node.t) / node.n) * -node.color


    exploration_score = math.sqrt(math.log(parent.n) / node.n)

    return node_score + (c / node.depth) * exploration_score

def pick_child_best_n(childs: list[Node]):
    if(childs is None or len(childs) == 0):
        return None
    
    best_node = childs[0]
    max_value = -math.inf

    for child in childs:
        value = child.n
        if(value > max_value):
            max_value = value
            best_node = child

    return best_node


def pick_child_best_ucb(childs: list[Node], c: float):
    best_node = childs[0]
    max_value = -math.inf

    for child in childs:
        value = get_ucb_score(child, c)
        if(value > max_value):
            max_value = value
            best_node = child

    return best_node

def pv_str(node: Node):
    if(node is None or node.parent is None):
        return ""
    
    return pv_str(node.parent) + "->" + move_str(node.move)

def mcts(board: np.array, color: float, model, iterations, c: float = 1, prior_weight = 2, verbose = False):
    root = Node(None, None, 0, False, color, None, None, 0)
    
    for _ in range(iterations):
        explore(root, c, model, prior_weight, board)

    # Choose the child with the largest number of visits:
    best_child = pick_child_best_n(root.childs)

    bc = root
    while bc is not None:
        if verbose and bc.childs is not None:
            for child in bc.childs:
                print(f"{pv_str(child)} (n: {child.n}) (t: {fstr(child.t)}) (own_reward: {round(child.own_reward * 100)}%) node_score: {fstr(get_ucb_score(child, 0))} (check: {child.is_check})")
    
        next_bc = pick_child_best_n(bc.childs)
        if(next_bc is None):
            print(f"pv: {pv_str(bc)}")
        
        bc = next_bc
    
    return best_child.move
