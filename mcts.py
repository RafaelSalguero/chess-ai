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

    def _expand(self):
        if(self.childs is not None):
            raise Exception("This node is already expanded")
        self.childs = get_childs(self)

    def explore(self, c: float, model):
        current = self

        # Search for the best child until we find a leaf node
        while current.childs is not None and len(current.childs) > 0:
            child_scores = [getUCBScore(child, c) for child in current.childs]
            max_score = max(child_scores)
            best_child_indices = [i for i, score in enumerate(child_scores) if score == max_score]
            best_child_index = random.choice(best_child_indices)
            current = current.childs[best_child_index]

        if current.n == 0:
            current.t += rollout(current, model)
            current._backprop(1)
        
        if current.childs is None:
            current._expand()

    def _backprop(self, n_inc):
        parent = self
        while parent.parent:
            parent = parent.parent
            parent.n += n_inc
            parent.t += self.t * parent.color * self.color
        
            

def get_childs(node: Node):
    moves = get_all_moves_slow(node.board, node.color)
    return list(map(lambda move: Node(node, apply_move(node.board, move), -node.color, move, node.depth + 1), moves))

@tf.function    
def internal_model_eval(model, x):
    return model(x)

def model_eval(model, color, board):
    b = board
    
    #y = internal_model_eval(model, onehot_encode_board(b).reshape((-1, 8, 8, 8))).numpy().reshape(-1)[0]

    #y = (y - 0.5) * 2 * color

    y = -evalBoard(b, color)


    print(y)
    print_board(board)
    return y

def rollout(node: Node, model):
    move_s =move_str(node.move) if node.move is not None else ""
    print(f"roll out depth {node.depth}, move: {move_s}, color: {node.color}")
    return model_eval(model, node.color, node.board)

def getUCBScore(node: Node, c: float):
    """
        Gets the UCB score, the tree search will explore nodes with higher scores first
    """
    if (node.n == 0):
        return float('inf')
    
    parent = node.parent if node.parent else node
    
    node_score = node.t / node.n
    exploration_score = math.sqrt(math.log(parent.n) / node.n)
    return node_score + c * exploration_score

def pick_best_child(node: Node):
    if(node.childs is None): 
        return None
    max_n = max(child.n for child in node.childs)
    max_childs = [child for child in node.childs if child.n == max_n]

    return random.choice(max_childs)

def mcts(board: np.array, color: float, model, iterations, c: float = 1):
    root = Node(None, board, color, None)
    for _ in range(iterations):
        root.explore(c, model)

    # Choose the child with the largest number of visits:
    best_child = pick_best_child(root)

    for child in root.childs:
        print(f"{move_str(child.move)} ({child.n})")

    print("pv: ")
    bc = best_child
    while (bc):
        print(move_str(bc.move) + "->", end="")
        bc = pick_best_child(bc)
    
    print("")

    return best_child.move
