import numpy as np
from eval import evalBoard, evalWin
from moves import apply_move, flip_board, get_all_moves, move_str_an
from test import get_test_boards
from minmax import minimax
from board import initialBoard
import os

from utils import get_np_hash, softmax
from view import print_board

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

def get_sim_games(eval_func = evalBoard, size=128, verbose=True):
    board = initialBoard
    
    added = set()
    ret = []
    y_evals = []

    color = 1

    eval = 0

    not_added_counter = 0

    prob_ratio = 1
    while True:

        train_board = board if color == 1 else flip_board(board)
        train_board_hash = get_np_hash(train_board)
        if(not train_board_hash in added):
            not_added_counter = 0
            added.add(train_board_hash)
            ret.append(train_board)
            y_evals.append(eval)
            if(len(ret) % 1000 == 0):
                print(f"sim_games count: {len(ret)}")
        else:
            not_added_counter += 1
            if(not_added_counter > 100):
                not_added_counter=0
                prob_ratio *= 0.99
                print(f'prob ratio: {prob_ratio}')

        if(verbose):
            print_board(board)
        if(len(ret) >= size):
            break

        moves = get_all_moves(board, color)

        if(len(moves)== 0):
            if(verbose):
                print(f'win: {evalWin(board)}')
            color = 1
            board = initialBoard
            continue
            
        next_boards = list(map(lambda move: apply_move(board, move), moves))
        next_color = -color
        next_evals = list(map(lambda board: eval_func(board, next_color), next_boards))

        if(verbose):
            print(next_evals)
        probs = softmax(-np.array(next_evals) * prob_ratio)
        best = np.random.choice(len(moves), size=1, p = probs)[0]

        if(verbose):
            moves_s = list(map(lambda move: move_str_an(board, moves, move), moves))
            print(moves_s)
            print(f'best: {moves_s[best]} ({next_evals[best]})')

        board = next_boards[best]
        color = next_color
        eval = next_evals[best]

    return (np.array(ret), np.array(y_evals))
    