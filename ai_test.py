from eval import evalDeadPosition
from minmax import iterative_deepening, variation_str, variation_str_an
from moves import allocate_moves_array
from view import parse_board, print_board

board = parse_board("""
8                        
7       ♝              ♚ 
6                        
5                        
4       ♔                
3                        
2                        
1                ♗       
  a  b  c  d  e  f  g  h
""")
print_board(board)
print(evalDeadPosition(board))
exit()
color = -1

depth = 3
print(f"depth: {depth}")
(next_eval, variation, iters, best_depth) = iterative_deepening(depth, 5, 2000, board, color, None, True, allocate_moves_array(), 0)

variation_text = variation_str_an(variation, board, color)
#variation_text  = variation_str(variation)
print(f"eval: {next_eval}, variation: {variation_text}, iters: {iters}")