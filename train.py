import numpy as np
from eval import evalBoard, evalWin
from moves import apply_move, apply_move_inplace, flip_board, get_all_moves, move_str_an, moves_str_an, undo_move_inplace
from test import get_test_boards
from minmax import iterative_deepening, minimax
from board import getInitialBoard, initialBoard
import os
from ttable import init_transposition_table

from utils import get_np_hash, softmax
from view import print_board
from numba import njit, prange

def get_test_board_train_data(size):
    print("generating test boards...")
    file_name= f"train_data/test_boards_{size}"

    if(os.path.isfile(file_name)):
        with np.load(file_name) as data:
            return data['boards']
    
    boards = np.array(get_test_boards(initialBoard, 1, 250, size))
    np.savez(file_name, boards=boards)
    return boards

def get_train_data(eval_func, size):
    x_train = get_test_board_train_data(size)
    print("evaluating test boards...")
    y_train = np.array(list(map(eval_func, x_train)))
    return (x_train, y_train)

def get_minmax_train_data(file_name, eval_func, tsize, vsize, depth, save_file = True):
    print("Generating test data for " + file_name + " size=" + str(tsize) + ", depth " + str(depth))
    file_name = f'train_data/{file_name}-d{depth}-t{tsize}-v{vsize}.npz'

    if(save_file and os.path.isfile(file_name)):
        with np.load(file_name) as data:
            print("Loading test data from file")

            n = np.histogram(data['y_train'], bins=163 * 2)
            np.savetxt("count.csv", n[0])
            return ((data['x_train'], data['y_train']), (data['x_test'], data['y_test']))
            
    
    def minmax_eval_board(board):
        (value,_,_) = minimax(board, 1, depth, eval_func)
        return value

    train = get_train_data(minmax_eval_board, tsize)
    test = get_train_data(minmax_eval_board, vsize)
    if(save_file):
        np.savez(file_name, x_train = train[0], y_train = train[1], x_test = test[0], y_test = test[1])
        
    return (train, test)

# Chooses a random index in probs where the probability is the value, the sum of probs should be 1
@njit
def random_choice(probs):
    rn = np.random.rand()
    min = 0
    for i, p in enumerate(probs):
        max = min + p
        if(rn >= min and rn < max):
            return i
        min = max
        
    return len(probs) - 1


@njit(parallel=True)
def get_sim_games(depth, max_iter, size=128, threads = 8, verbose=True):
    if(size % threads != 0):
        raise Exception("Size must be a multiple of threads")
    
    ttable = init_transposition_table(1024 * 1024 * 1024)

    boards = np.empty((size, 8, 8), dtype=np.int32)
    y_evals = np.empty(size)

    step = size // threads
    
    for thread in prange(threads):
        get_sim_games_inplace(depth, max_iter, boards, y_evals, thread * step, ttable, f'{thread}/{threads} - ', step, verbose)

    return (boards, y_evals)
@njit
def get_sim_games_inplace(depth, max_iter, dest_boards, dest_evals, dest_index, ttable, prefix = '', size=128, verbose=True):
    initial_copy = np.copy(initialBoard)
    board = np.copy(initial_copy)
    
    added = set()
    color = 1
    eval = 0

    not_added_counter = 0

    prob_ratio = 2

    index = 0

    while True:
        train_board = board if color == 1 else flip_board(board)
        train_board_hash = get_np_hash(train_board)
        if(not train_board_hash in added):
            not_added_counter = 0
            added.add(train_board_hash)
            dest_boards[index + dest_index] = train_board
            dest_evals[index + dest_index] = eval
            index += 1

            if(index >= size):
                break

            print(f"{prefix}sim_games count: {index}")
        else:
            not_added_counter += 1
            if(not_added_counter > 100):
                not_added_counter=0
                prob_ratio *= 0.99
                print(f'{prefix}prob ratio: ' + str(int(round(prob_ratio * 100))))

        if(verbose):
            print_board(board)

        moves = get_all_moves(board, color)
        if(len(moves)== 0):
            if(verbose):
                print(f'win: {evalWin(board)}')
            color = 1
            board = initial_copy
            continue
        
        next_evals = []
        next_color = -color

        for move in moves:
            undo = apply_move_inplace(board, move)
            next_evals.append(iterative_deepening(depth, max_iter, board, next_color, ttable))
            undo_move_inplace(board, move, undo)

        if(verbose):
            print(next_evals)
        probs = softmax(-np.array(next_evals) * prob_ratio)
        best = random_choice(probs)

        if(verbose):
            moves_s = moves_str_an(board, moves)
            print(moves_s)
            print(f'{prefix}best: {moves_s[best]} ({next_evals[best]})')

        board = apply_move(board, moves[best])

        color = next_color
        eval = next_evals[best]