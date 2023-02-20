import numpy as np
from numba import njit, vectorize, float32, int32
def softmax(x):
    return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())

hash_randoms = np.random.default_rng(123456).integers(0, 65535, size=1024).astype(dtype=np.int32)

@vectorize([int32(int32, int32)])
def add_mod_2_16(x, y):
    return (x + y ) % 65536

def get_np_hash(arr):
    if(arr.dtype != np.int32):
        raise Exception("Type of arr should be int32")
    
    arr = arr.reshape(-1)
    len = arr.shape[0]
    return add_mod_2_16.reduce(arr * hash_randoms[0:len])


_onehot_diags = np.concatenate((-np.flip(np.eye(8), 0), np.zeros((1,8)), np.eye(8)))
def onehot_encode_board(x):
    return _onehot_diags[x]
