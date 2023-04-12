import math
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


_onehot_diags = np.concatenate((-np.flip(np.eye(8), 0), np.zeros((1,8)), np.eye(8))).astype(np.float32)

def onehot_encode_board(x):
    return _onehot_diags[x + 8]

@njit
def onehot_encode_board_n(board, dest):
    for y in range(0, 8):
        for x in range(0, 8):
            piece = board[y, x]
            for i in range(0, 8):
                dest[y, x, i] = 0
            if(piece != 0):
                index = abs(piece) - 1
                dest[y, x, index] = np.sign(piece)
            

@njit
def shuffle(items):
    i = len(items)
    while i > 1:
        i = i - 1
        j = np.random.randint(i)  # 0 <= j <= i-1
        items[j], items[i] = items[i], items[j]


@njit
def cap_histogram(y, cap, avg_size):
    """
        Probabilistically caps the count per bin average on the histogram of y,
        filtering out more frequent values of y

        Returns the picked indices from y
    """
    min_y = np.min(y)
    bins = np.arange(-150, 152, 1)
    (h, bins) = np.histogram(y, bins)
    probs = np.zeros(len(h))
    
    r_avg = 0
    total_aprox = 0
    for i in range(0, len(h)):
        tail_i = i - avg_size
        tail = 0 if tail_i < 0 else h[tail_i]
        curr = h[i]
        r_avg = ((r_avg * avg_size - tail) + curr) / avg_size
        probs[i] = min(cap /(r_avg + 0.01),1)
        total_aprox += curr * probs[i]

    indices = np.empty(round(total_aprox), np.int64)
    out_index = 0
    y_index = 0
    while out_index < len(indices):
        value = y[y_index]
        bin = round(y[y_index] - min_y)
        prob = probs[bin]
        if(np.random.uniform(0, 1) < prob):
            indices[out_index] = y_index
            out_index += 1
        
        y_index = (y_index + 1) % len(y)
        
    return indices

@njit
def cut_trail(f_str):
    cut = 0
    for c in f_str[::-1]:
        if c == "0":
            cut += 1
        else:
            break
    if cut == 0:
        for c in f_str[::-1]:
            if c == "9":
                cut += 1
            else:
                cut -= 1
                break
    if cut > 0:
        f_str = f_str[:-cut]
    if f_str == "":
        f_str = "0"
    return f_str


@njit
def fstr(value):
    if math.isnan(value):
        return "nan"
    elif value == 0.0:
        return "0.0"
    elif value < 0.0:
        return "-" + fstr(-value)
    elif math.isinf(value):
        return "inf"
    else:
        max_digits = 16
        min_digits = -4
        e10 = math.floor(math.log10(value)) if value != 0.0 else 0
        if min_digits < e10 < max_digits:
            i_part = math.floor(value)
            f_part = math.floor((1 + value % 1) * 10.0 ** max_digits)
            i_str = str(i_part)
            f_str = cut_trail(str(f_part)[1:max_digits - e10])
            return i_str + "." + f_str
        else:
            m10 = value / 10.0 ** e10
            exp_str_len = 4
            i_part = math.floor(m10)
            f_part = math.floor((1 + m10 % 1) * 10.0 ** max_digits)
            i_str = str(i_part)
            f_str = cut_trail(str(f_part)[1:max_digits])
            e_str = str(e10)
            if e10 >= 0:
                e_str = "+" + e_str
            return i_str + "." + f_str + "e" + e_str