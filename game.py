from board import initialBoard, testInitialBoard
from chess import find_best_move
from view import print_board
from chess import ai_eval_board, find_best_move
from moves import move_str, flip_board, flip_move, get_all_moves, str_move, apply_move
from minmax import minimax
from eval import evalBoard, win_threshold
import numpy as np

def console_player(board, white):
    all_moves = get_all_moves(board)
    
    valid_next_moves_str = list(map(lambda move: move_str(move if white == 1 else flip_move(move)), all_moves))
    print(valid_next_moves_str)

    print(("white " if white==1 else "black") +  " - next move?")

    next_move = ''
    while True:
        next_move = input()
        if(next_move in valid_next_moves_str):
            next_move = str_move(next_move)
            break

        print("invalid move")
    if(white == -1):
        next_move = flip_move(next_move)
    
    return next_move

def ai_player(board, white):
    print("ai is thinking")
    best_move = find_best_move(board)
    print("ai best move: " + move_str(best_move))

    return best_move

def play(board, white, white_player, black_player):
    while True:
        print_board(board)

        if(white == -1):
            board = flip_board(board)

        eval_depths = [0, 1, 2, 3, 4]
        evals = np.array(list(map(lambda depth: minimax(board, depth, evalBoard, win_threshold), eval_depths))) * white
        print("minmax eval", evals)
        # print("ai eval", ai_eval_board(board))

        moves = get_all_moves(board)

        if(len(moves) == 0):
            print("No more moves")
            break
        
        player = white_player if white==1 else black_player
        next_move = player(board, white)

        board = apply_move(board, next_move)

        if(white == -1):
            board = flip_board(board)
            
        white = -white

play(testInitialBoard, 1, ai_player, ai_player)