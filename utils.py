import numpy as np
from numba import njit, vectorize, float32, int32


@njit
def softmax(x):
    return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())

hash_randoms = np.random.default_rng(123456).integers(0, 9223372036854775807, size=1024).astype(dtype=np.int64)
@njit
def get_np_hash(arr):
    arr = (arr).reshape(-1)
    len = arr.shape[0]
    return np.abs(np.int64(np.sum((arr + 100) * hash_randoms[0:len])))


_onehot_diags = np.concatenate((-np.flip(np.eye(8), 0), np.zeros((1,8)), np.eye(8)))
def onehot_encode_board(x):
    return _onehot_diags[x + 8]

@njit
def shuffle(items):
    i = len(items)
    while i > 1:
        i = i - 1
        j = np.random.randint(i)  # 0 <= j <= i-1
        items[j], items[i] = items[i], items[j]
