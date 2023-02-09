import numpy as np
from minmax import minimax
from eval import win_threshold
from board import get_test_random_board
import os

def get_train_data(get_random_game, eval_func, size):
    x_train = np.array(list(map(lambda x: get_random_game(), range(0, size))))
    y_train = np.array(list(map(eval_func, x_train)))
    return (x_train, y_train)

def get_minmax_train_data(file_name, get_random_game, eval_func, tsize, vsize, depth):
    print("Generating test data for " + file_name + " size=" + str(tsize) + ", depth " + str(depth))
    file_name = "train_data_" + file_name + "-d" + str(depth) + ".npz"

    if(os.path.isfile(file_name)):
        with np.load(file_name) as data:
            print("Loading test data from file")
            return ((data['x_train'], data['y_train']), (data['x_test'], data['y_test']))
            
    
    def minmax_eval_board(board):
        return minimax(board, 1, depth, eval_func, win_threshold)

    train = get_train_data( get_random_game, minmax_eval_board, tsize)
    test = get_train_data( get_random_game, minmax_eval_board, vsize)
    np.savez(file_name, x_train = train[0], y_train = train[1], x_test = test[0], y_test = test[1])
    return (train, test)

