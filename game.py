from board import initialBoard
from chess import find_best_move
from view import print_board
from moves import move_str, flip_board, flip_move, get_all_moves, str_move, apply_move
board = initialBoard
white = True

def console_player(board, white):
    all_moves = get_all_moves(board)
    
    valid_next_moves_str = list(map(lambda move: move_str(move if white else flip_move(move)), all_moves))
    print(valid_next_moves_str)

    print("next move?")

    next_move = ''
    while True:
        next_move = input()
        if(next_move in valid_next_moves_str):
            next_move = str_move(next_move)
            break

        print("invalid move")
    if(not white):
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
        if(not white):
            board = flip_board(board)

        player = white_player if white else black_player
        next_move = player(board, white)

        board = apply_move(board, next_move)

        if(not white):
            board = flip_board(board)
            
        white = not white

play(initialBoard, True, console_player, ai_player)