from eval import evalBoard
from minmax import minimax_eval_board
from train import get_sim_games
from view import parse_board, print_board

from timeit import timeit
(x_train, y_train) = get_sim_games(minimax_eval_board(100, 5, evalBoard), size=1024, verbose= False)

for i, x in enumerate(x_train):
    y = y_train[i]

    print_board(x)
    print(f'eval: {y}')