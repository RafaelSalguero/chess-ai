from board import initialBoard, testInitialBoard, errBoard
from view import print_board
from chess import ai_eval_board, find_best_move
from moves import move_str, flip_board, flip_move, get_all_moves, str_move, apply_move
from minmax import minimax
from eval import evalBoard, win_threshold, evalWin
import numpy as np

def console_player(board, color):
    all_moves = get_all_moves(board, color)
    
    valid_next_moves_str = list(map(move_str, all_moves))
    print(valid_next_moves_str)

    print(("white " if color==1 else "black") +  " - next move?")

    next_move = ''
    while True:
        next_move = input()
        if(next_move in valid_next_moves_str):
            next_move = str_move(next_move)
            break

        print("invalid move")
    
    return next_move

def ai_player(board, color):
    print("ai is thinking")
    if(color == -1):
        board = flip_board(board)
    
    best_move = find_best_move(board)

    if(color == -1):
        best_move = flip_move(best_move)

    print("ai best move: " + move_str(best_move))

    return best_move

def play(board, color, white_player, black_player):
    while True:
        print_board(board)

        win = evalWin(board)
        if(win != 0):
            print("Win: " + ("white" if win == 1 else "black"))
            break

        eval_depths = [0, 1, 2, 3, 4]
        evals = np.array(list(map(lambda depth: minimax(board, color, depth, evalBoard, win_threshold), eval_depths)))
        print("minmax eval", evals)
        print("ai eval", ai_eval_board(board))

        moves = get_all_moves(board, color)

        if(len(moves) == 0):
            print("No more moves")
            break
        
        player = white_player if color==1 else black_player
        next_move = player(board, color)

        board = apply_move(board, next_move)

        color = -color

play(testInitialBoard, 1, console_player, ai_player)