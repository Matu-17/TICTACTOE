"""Pruebas unitarias para los agentes Minimax y Árbol Binario de Decisión (ML)."""

import os
import unittest
from model.game_model import GameModel, GameState
from model.minimax_agent import MinimaxAgent
from model.ml_agent import BinaryDecisionTree, MLAgent


class TestMinimaxAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.model = GameModel()
        self.agent = MinimaxAgent(ai_player="O", human_player="X", difficulty=MinimaxAgent.DIFFICULTY_HARD)

    def test_minimax_takes_immediate_winning_move(self) -> None:
        # Board:
        # O O . (positions 0, 1, 2)
        # X X . (positions 3, 4, 5)
        # . . . (positions 6, 7, 8)
        self.model.make_move(3)  # X at 3
        self.model.make_move(0)  # O at 0
        self.model.make_move(4)  # X at 4
        self.model.make_move(1)  # O at 1
        # Now it is X's turn to play at 5, but let's test if O can find winning move at 2:
        self.model._current_player = "O"
        best_move = self.agent.get_best_move(self.model)
        self.assertEqual(best_move, 2)

    def test_minimax_blocks_immediate_opponent_win(self) -> None:
        # Board:
        # X X . (positions 0, 1, 2)
        # . O . (positions 3, 4, 5)
        # . . . (positions 6, 7, 8)
        self.model.make_move(0)  # X
        self.model.make_move(4)  # O
        self.model.make_move(1)  # X
        # O must block X by playing at 2
        best_move = self.agent.get_best_move(self.model)
        self.assertEqual(best_move, 2)

    def test_minimax_is_unbeatable_against_random_player(self) -> None:
        """Verifica que Minimax en dificultad Hard nunca pierda (0 derrotas en 30 partidas)."""
        import random
        for _ in range(30):
            sim = GameModel()
            ai = MinimaxAgent(ai_player="O", human_player="X", difficulty=MinimaxAgent.DIFFICULTY_HARD)
            
            while not sim.is_game_over():
                if sim.get_current_player() == "X":
                    # Jugador humano aleatorio
                    move = random.choice(sim.get_available_moves())
                    sim.make_move(move)
                else:
                    # IA Minimax
                    move = ai.get_best_move(sim)
                    self.assertIsNotNone(move)
                    sim.make_move(move)

            # La IA 'O' nunca debe perder
            self.assertNotEqual(sim.get_state(), GameState.X_WON)


class TestMLAgentAndDecisionTree(unittest.TestCase):
    def setUp(self) -> None:
        self.ml_agent = MLAgent(ai_player="O", dataset_path="data/test_dataset.csv")

    def tearDown(self) -> None:
        if os.path.exists("data/test_dataset.csv"):
            os.remove("data/test_dataset.csv")

    def test_board_encoding(self) -> None:
        board = ["X", "O", "", "", "X", "O", "", "", ""]
        encoded = MLAgent.encode_board(board)
        self.assertEqual(encoded, [1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0])

    def test_binary_decision_tree_fit_and_predict(self) -> None:
        tree = BinaryDecisionTree(max_depth=5)
        # Muestras sintéticas simples
        X = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
        y = [0, 1, 2, 0]
        tree.fit(X, y)
        self.assertGreater(tree.accuracy, 50.0)

        preds = tree.predict(X)
        self.assertEqual(len(preds), 4)

    def test_dataset_generation_and_training(self) -> None:
        # Generar dataset de prueba con 30 partidas
        csv_path = self.ml_agent.generate_dataset(num_games=30)
        self.assertTrue(os.path.exists(csv_path))

        self.ml_agent.train()
        self.assertTrue(self.ml_agent.is_trained)
        self.assertGreater(self.ml_agent.tree.accuracy, 70.0)

        # Inferencia con modelo
        model = GameModel()
        move = self.ml_agent.get_best_move(model)
        self.assertIn(move, list(range(9)))


if __name__ == "__main__":
    unittest.main()
