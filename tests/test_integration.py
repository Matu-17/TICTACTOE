"""Pruebas de integración para Controller, Model y Agentes IA."""

import unittest
from unittest.mock import MagicMock
from model.game_model import GameModel, GameState
from controller.game_controller import GameController


class TestIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.model = GameModel()
        self.mock_view = MagicMock()
        self.mock_view.root = MagicMock()
        self.controller = GameController(self.model, self.mock_view)

    def test_controller_initialization_sets_callbacks_and_refreshes_view(self) -> None:
        self.mock_view.set_callbacks.assert_called_once()
        self.mock_view.update_board.assert_called_with([""] * 9, None)
        self.mock_view.update_status.assert_called_with("Turno actual: Jugador X")

    def test_controller_handle_cell_click_advances_game(self) -> None:
        self.controller.handle_cell_click(0)
        self.assertEqual(self.model.get_board()[0], "X")
        self.assertEqual(self.model.get_current_player(), "O")
        self.mock_view.update_status.assert_called_with("Turno actual: Jugador O")

    def test_controller_handle_win_sequence(self) -> None:
        # X: 0, O: 3, X: 1, O: 4, X: 2
        for move in [0, 3, 1, 4, 2]:
            self.controller.handle_cell_click(move)

        self.assertEqual(self.model.get_state(), GameState.X_WON)
        self.assertEqual(self.model.get_winning_line(), (0, 1, 2))
        self.mock_view.update_board.assert_called_with(
            self.model.get_board(), (0, 1, 2)
        )
        self.mock_view.update_status.assert_called_with(
            "🎉 ¡Victoria del Jugador X!", color="#27AE60"
        )

        # Clicks after game over should be ignored
        self.controller.handle_cell_click(8)
        self.assertEqual(self.model.get_board()[8], "")

    def test_controller_reset_game(self) -> None:
        self.controller.handle_cell_click(0)
        self.controller.handle_reset_game()
        self.assertEqual(self.model.get_board(), [""] * 9)
        self.assertEqual(self.model.get_current_player(), "X")
        self.assertEqual(self.model.get_state(), GameState.PLAYING)

    def test_controller_switch_to_minimax_mode(self) -> None:
        self.controller.handle_mode_change(GameController.MODE_HUMAN_VS_MINIMAX)
        self.assertEqual(self.controller.current_mode, GameController.MODE_HUMAN_VS_MINIMAX)
        # Humano juega casilla 0
        self.controller.handle_cell_click(0)
        self.assertEqual(self.model.get_board()[0], "X")
        # El controlador programa la respuesta de la IA
        self.mock_view.root.after.assert_called_with(300, self.controller._execute_ai_turn)

    def test_controller_switch_to_ml_mode(self) -> None:
        self.controller.handle_mode_change(GameController.MODE_HUMAN_VS_ML)
        self.assertEqual(self.controller.current_mode, GameController.MODE_HUMAN_VS_ML)
        # Humano juega casilla 4 (centro)
        self.controller.handle_cell_click(4)
        self.assertEqual(self.model.get_board()[4], "X")
        # El controlador programa la respuesta de la IA
        self.mock_view.root.after.assert_called_with(300, self.controller._execute_ai_turn)

    def test_controller_symbol_selection(self) -> None:
        # Humano elige 'O' (segundo turno)
        self.controller.handle_mode_change(GameController.MODE_HUMAN_VS_MINIMAX)
        self.controller.handle_symbol_change("O")
        self.assertEqual(self.controller.human_symbol, "O")
        self.assertEqual(self.controller.ai_symbol, "X")
        # Como X mueve primero y la IA es X, la IA debe mover inmediatamente
        self.mock_view.root.after.assert_called_with(350, self.controller._execute_ai_turn)


if __name__ == "__main__":
    unittest.main()
