import random
import numpy as np
import math
from eval import evalBoard, evalWin
from moves import apply_move, flip_board, get_all_moves_slow, move_str
from utils import onehot_encode_board
from view import print_board
import tensorflow as tf

class Node:
    def __init__(self, parent, board: np.array, color: float = 1, move: np.array = None, depth = 0):
        self.parent = parent
        self.board = board
        self.childs = None
        self.move = move
        self.color = color
        self.depth = depth
        
        # total win count
        self.t = 0
        # visit count
        self.n = 0
        self.prior = 0
        self.own_reward = 0
        self.is_checkmate = evalWin(board) == -color
        self.is_check = False

    def _expand(self, color, model, prior_weight):
        if(self.childs is not None):
            raise Exception("This node is already expanded")
        self.childs = get_childs(self, color, model, prior_weight)

    def remove_check(self):
        if(self.parent is None):
            return
        self.parent.childs.remove(self)
        backprop(self.parent, -self.n, -self.t)

        if(len(self.parent.childs) == 0):
            # No more moves in parent, so this is a checkmate
            self.own_reward = 1.0


    def explore(self, c: float, model, prior_weight):
        current = self

        # Search for the best child until we find a leaf node
        while current.childs is not None and len(current.childs) > 0:
            child_scores = [getUCBScore(child, c) for child in current.childs]
            max_score = max(child_scores)
            best_child_indices = [i for i, score in enumerate(child_scores) if score == max_score]
            best_child_index = random.choice(best_child_indices)
            current = current.childs[best_child_index]

        if current.n > 0 and current.childs is None:
            current._expand(self.color, model, prior_weight)
            if(current.is_check):
                current.remove_check()
                return
            
            if len(current.childs) > 0:
                current = random.choice(current.childs)

        reward = current.own_reward #rollout(current, self.color, model)
        current.own_reward = reward
        backprop(current, 1, reward)

        
            
def backprop(node: Node, n, t):
    parent = node
    while parent:
        parent.n += n
        parent.t += t
        parent = parent.parent
    
def get_childs(node: Node, color, model, prior_weight):
    moves = get_all_moves_slow(node.board, node.color)

    def create_node(move):
        next_board = apply_move(node.board, move)
        ret = Node(node, next_board, -node.color, move, node.depth + 1)
        ret.own_reward = rollout(ret, color, model)

        # prior probability:
        ret.prior = ret.own_reward * prior_weight
        if ret.is_checkmate:
            node.is_check = True

        return ret
    
    return list(map(create_node, moves))

@tf.function    
def internal_model_eval(model, x):
    return model(x)

def from_model_space(x):
    return np.power((x - 0.5) * (2 * np.cbrt(150)), 3)

def eval_to_prob(x):
    """
        Maps from [-150, 150] to [-1, 1]
    """
    return x / 150

def model_eval(model, color, board):
    b = board
    
    if(color == -1):
        b = flip_board(board)
    
    nn_eval = True
    y = 0
    if(nn_eval):
        y = internal_model_eval(model, onehot_encode_board(b).reshape((-1, 8, 8, 8))).numpy().reshape(-1)[0]
        y = (y - 0.5) * 2
    else:
        y = eval_to_prob(evalBoard(board, 1))

    return y

def rollout(node: Node, color, model):
    win_val = evalWin(node.board) * 1.05

    if(win_val == 0):
        y = model_eval(model, color, node.board)
    else:
        y = win_val * color

    return y

def getUCBScore(node: Node, c: float):
    """
        Gets the UCB score, the tree search will explore nodes with higher scores first
    """
    if (node.n == 0):
        return float('inf')
    
    parent = node.parent if node.parent else node
    
    if (parent.n == 0):
        return float('inf')
    
    node_score = ((node.prior + node.t) / node.n) * -node.color
    if(parent.n < 1):
        print(f"parent {pv_str(parent)} n is {parent.n}")

    if(node.n < 1):
        print(f"node {pv_str(node)} n is {node.n}")

    exploration_score = math.sqrt(math.log(parent.n) / node.n)
    return node_score + (c / node.depth) * exploration_score

def pick_best_child(node: Node):
    if(node.childs is None or len(node.childs) == 0): 
        return None
    max_n = max(child.n for child in node.childs)
    max_childs = [child for child in node.childs if child.n == max_n]

    return random.choice(max_childs)

def pv_str(node: Node):
    if(node is None or node.parent is None):
        return ""
    
    return pv_str(node.parent) + "->" + move_str(node.move)

def mcts(board: np.array, color: float, model, iterations, c: float = 1, prior_weight = 2, verbose = False):
    root = Node(None, board, color, None)
    for _ in range(iterations):
        root.explore(c, model, prior_weight)

    # Choose the child with the largest number of visits:
    best_child = pick_best_child(root)

    bc = root
    while (bc):
        if verbose and bc.childs:
            for child in bc.childs:
                print(f"{pv_str(child)} (n: {child.n}) (t: {child.t}) (own_reward: {round(child.own_reward * 100)}%) node_score: {getUCBScore(child, 0)} (check: {child.is_check})")
    
        next_bc = pick_best_child(bc)
        if(next_bc is None):
            print(f"pv: {pv_str(bc)}")
        
        bc = next_bc
    
    return best_child.move
