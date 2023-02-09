from board import initialBoard, testInitialBoard
from view import print_board
from ai import ai_eval_board, find_best_move_ai, find_best_move_minimax
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

def auto_player(board, color, find_best_move):
    # print("ai is thinking")
    if(color == -1):
        board = flip_board(board)
    
    best_move = find_best_move(board)

    if(color == -1):
        best_move = flip_move(best_move)

    # print("ai best move: " + move_str(best_move))

    return best_move

def ai_player(board, color):
    return auto_player(board, color, find_best_move_ai)

def minimax_player(depth):
    def player(board, color):
        return auto_player(board, color, lambda board: find_best_move_minimax(board, depth))
    
    return player

def play(board, color, white_player, black_player, print_evals = True,show_board = True, verbose = True, max_depth = 100):
    while True:
        max_depth -= 1
        if(max_depth <= 0):
            print("max depth reached")
            return 0
        
        if(show_board):
            print_board(board)

        win = evalWin(board)
        if(win != 0):
            print("Win: " + ("white" if win == 1 else "black"))
            return win

        if(print_evals):
            eval_depths = [0, 1, 2, 3, 4]
            evals = np.array(list(map(lambda depth: minimax(board, color, depth, evalBoard, win_threshold), eval_depths)))
            print("minmax eval", evals)
            print("ai eval", ai_eval_board(board))

        moves = get_all_moves(board, color)

        if(len(moves) == 0):
            print("No more moves")
            return 0
        
        player = white_player if color==1 else black_player
        next_move = player(board, color)

        board = apply_move(board, next_move)

        color = -color

# Run N games between a and b playes, returns the win rate for a player a
def simulateGames(board, a, b, count):
    color = 1
    draws = 0
    score = 0

    a_wins = 0
    b_wins = 0
    a_rate = 0
    for i in range(0, count):
        print(f'Simulating game {i}/{count}')
        win = play(board, 1, a, b, False, False, False) * color
        
        if(win == 0):
            draws+=1
        else:
            score += win

        # flip:
        x = a
        a = b
        b = x
        color = -color

        # calc:
        total = i + 1
        a_wins = (total + score) / 2
        b_wins = total - a_wins
        a_rate = a_wins / total

        print(f'a_wins: {a_wins}, b_wins: {b_wins}, a_rate: {a_rate}, draws: {draws}, score: {score}')

  
    return a_rate

simulateGames(testInitialBoard, minimax_player(2), minimax_player(3), 100)