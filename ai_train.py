import os
import numpy as np
from eval import evalWin
from train import get_sim_games
from numba import njit


def get_train_data(depth, quiescence_depth, max_iter, size, cache = False):
    file_name = f'train_data/get_sim_games_{depth}_{max_iter}_{size}.npz'
    if(cache and os.path.isfile(file_name)):
        with np.load(file_name) as data:
            return (data['x'], data['y'], data['ply'])
        
    (x, y, ply) = get_sim_games(depth, quiescence_depth, max_iter, size=size, threads=8, verbose=False)

    if (cache):
        np.savez_compressed(file_name, x=x, y=y, ply=ply)
    return (x, y, ply)

@njit
def reduce_non_wins(x_train, y_train):
    for i, x in enumerate(x_train):
        if(evalWin(x) == 0):
            y_train[i] = min(max(y_train[i], -120), 120)