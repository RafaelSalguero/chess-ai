from view import print_board
from moves import move_str, flip_board, flip_move, get_all_moves, move_str_an, apply_move, str_move_an
from minmax import minimax, find_best_move_minimax
from eval import evalBoard, evalWin
import numpy as np

def console_player(board, color):
    all_moves = get_all_moves(board, color)
    
    valid_next_moves_str = list(map(lambda move: move_str_an(board, all_moves, move), all_moves))
    print(valid_next_moves_str)

    print(("white " if color==1 else "black") +  " - next move?")

    next_move = ''
    while True:
        next_move = input()
        if(next_move in valid_next_moves_str):
            next_move = str_move_an(board, all_moves, next_move)
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


def minimax_player(depth, eval_func = evalBoard):
    def player(board, color):
        return auto_player(board, color, lambda board: find_best_move_minimax(board, depth, eval_func))
    
    return player

def play(board, color, white_player, black_player, print_evals = True,show_board = True, verbose = True, max_depth = 500):
    while True:
        max_depth -= 1
        if(max_depth <= 0):
            return 0
        
        if(show_board):
            print_board(board)

        win = evalWin(board)
        if(win != 0):
            return win

        if(print_evals):
            eval_depths = [0, 1, 2, 3]
            evals = np.array(list(map(lambda depth: minimax(board, color, depth, evalBoard)[0], eval_depths)))
            print("minmax eval", evals)

        moves = get_all_moves(board, color)

        if(len(moves) == 0):
            print("No more moves")
            return 0
        
        player = white_player if color==1 else black_player
        next_move = player(board, color)

        if(show_board):
            print(move_str_an(board, moves, next_move))

        board = apply_move(board, next_move)

        color = -color

# Run N games between a and b playes, returns the win rate for a player a
def simulateGames(board, a, b, count, print_games = False, print_messages = True):
    color = 1
    draws = 0
    score = 0

    a_wins = 0
    b_wins = 0
    a_rate = 0
    for i in range(0, count):
        player_color_msg = 'white' if color == 1 else 'black'
        if(print_messages):
            print(f'Simulating game {i}/{count}, a = {player_color_msg}')
        win = play(board, 1, a, b, False, print_games, False) * color
        
        if(win == 0):
            draws+=1
        else:
            winner = "a" if win == 1 else "b"
            winner_color =  "white" if win * color == 1 else "black"
            if(print_messages):
                print(f'Winner: {winner} ({winner_color})')
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

        if(print_messages):
            print(f'a_wins: {a_wins}, b_wins: {b_wins}, a_rate: {a_rate}, draws: {draws}, score: {score}')

  
    return a_rate

# play(initialBoard, 1, console_player, console_player)