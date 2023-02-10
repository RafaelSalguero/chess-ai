from game import minimax_player
from moves import apply_move, flip_board, move_str
from view import parse_board, print_board, black_chars
from ai import ai_player, curr_model, ai_eval_board

board = parse_board('''
8                        
7                        
6             ♚          
5                        
4    ♙  ♔                
3                        
2                   ♟︎    
1                        
  a  b  c  d  e  f  g  h 
''')

print_board(board)

player = ai_player(curr_model, True)

next_move = player(board, 1)
print(move_str(next_move))

next_board = flip_board(apply_move(board, next_move))

print_board(next_board)
print("eval", ai_eval_board(curr_model, next_board))