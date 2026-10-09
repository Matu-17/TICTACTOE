"""Pruebas unitarias para el modelo del juego (GameModel)."""

import unittest
from model.game_model import GameModel, GameState


class TestGameModel(unittest.TestCase):
    def setUp(self) -> None:
        self.model = GameModel()

    def test_initial_board_empty(self) -> None:
        self.assertEqual(self.model.get_board(), [""] * 9)
        self.assertEqual(self.model.get_current_player(), "X")
        self.assertEqual(self.model.get_state(), GameState.PLAYING)
        self.assertEqual(len(self.model.get_available_moves()), 9)
        self.assertFalse(self.model.is_game_over())

    def test_valid_move(self) -> None:
        result = self.model.make_move(0)
        self.assertTrue(result)
        self.assertEqual(self.model.get_board()[0], "X")
        self.assertEqual(self.model.get_current_player(), "O")
        self.assertEqual(len(self.model.get_available_moves()), 8)

    def test_invalid_move_occupied_cell(self) -> None:
        self.model.make_move(0)  # X plays at 0
        result = self.model.make_move(0)  # O tries to play at 0
        self.assertFalse(result)
        self.assertEqual(self.model.get_current_player(), "O")

    def test_invalid_move_out_of_bounds(self) -> None:
        self.assertFalse(self.model.is_valid_move(-1))
        self.assertFalse(self.model.is_valid_move(9))
        self.assertFalse(self.model.make_move(10))

    def test_horizontal_win(self) -> None:
        # X: 0, O: 3, X: 1, O: 4, X: 2 -> X wins row 0
        self.model.make_move(0)  # X
        self.model.make_move(3)  # O
        self.model.make_move(1)  # X
        self.model.make_move(4)  # O
        self.model.make_move(2)  # X
        self.assertEqual(self.model.get_state(), GameState.X_WON)
        self.assertEqual(self.model.check_winner(), "X")
        self.assertEqual(self.model.get_winning_line(), (0, 1, 2))
        self.assertTrue(self.model.is_game_over())
        # Movements blocked after game over
        self.assertFalse(self.model.make_move(5))

    def test_vertical_win(self) -> None:
        # X: 0, O: 1, X: 3, O: 4, X: 6 -> X wins col 0
        self.model.make_move(0)  # X
        self.model.make_move(1)  # O
        self.model.make_move(3)  # X
        self.model.make_move(4)  # O
        self.model.make_move(6)  # X
        self.assertEqual(self.model.get_state(), GameState.X_WON)
        self.assertEqual(self.model.get_winning_line(), (0, 3, 6))

    def test_diagonal_win(self) -> None:
        # X: 0, O: 1, X: 4, O: 2, X: 8 -> X wins main diagonal
        self.model.make_move(0)  # X
        self.model.make_move(1)  # O
        self.model.make_move(4)  # X
        self.model.make_move(2)  # O
        self.model.make_move(8)  # X
        self.assertEqual(self.model.get_state(), GameState.X_WON)
        self.assertEqual(self.model.get_winning_line(), (0, 4, 8))

    def test_draw(self) -> None:
        # Board:
        # X O X
        # X X O
        # O X O
        moves = [0, 1, 2, 4, 3, 5, 7, 6, 8]
        for m in moves:
            self.model.make_move(m)
        self.assertEqual(self.model.get_state(), GameState.DRAW)
        self.assertTrue(self.model.is_draw())
        self.assertIsNone(self.model.check_winner())
        self.assertTrue(self.model.is_game_over())

    def test_reset_game(self) -> None:
        self.model.make_move(0)
        self.model.make_move(1)
        self.model.reset_game()
        self.assertEqual(self.model.get_board(), [""] * 9)
        self.assertEqual(self.model.get_current_player(), "X")
        self.assertEqual(self.model.get_state(), GameState.PLAYING)
        self.assertIsNone(self.model.get_winning_line())

    def test_undo_move(self) -> None:
        self.model.make_move(4)
        self.assertEqual(self.model.get_board()[4], "X")
        self.assertEqual(self.model.get_current_player(), "O")

        success = self.model.undo_move(4)
        self.assertTrue(success)
        self.assertEqual(self.model.get_board()[4], "")
        self.assertEqual(self.model.get_current_player(), "X")
        self.assertEqual(self.model.get_state(), GameState.PLAYING)


if __name__ == "__main__":
    unittest.main()
