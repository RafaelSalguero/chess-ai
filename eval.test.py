import unittest
from board import emptyCell, rook, knight, bishop, queen, pawn, king, initialBoard, piecesRank, pawnsRank
from eval import evalBoard, evalCell

class TestEval(unittest.TestCase):
    def test_pieces(self):
        self.assertEqual(evalCell(pawn), 1)
        self.assertEqual(evalCell(knight), 3)
        self.assertEqual(evalCell(bishop), 3)
        self.assertEqual(evalCell(queen), 9)
        self.assertEqual(evalCell(king), 50)
        self.assertEqual(evalCell(rook), 5)
        self.assertEqual(evalCell(emptyCell), 0)

    def test_initialBoard(self):
        self.assertEqual(evalBoard(initialBoard), 0)

    def test_rows(self):
        self.assertEqual(evalBoard(piecesRank), 81)
        self.assertEqual(evalBoard(pawnsRank), 8)

if __name__ == '__main__':
    unittest.main()