"""Módulo del Agente Minimax (Semana 2).

Implementa el algoritmo Minimax clásico con Backtracking recursivo sobre el tablero
de Tres en Raya, con soporte para contador de nodos explorados, medición de latencia
y tres niveles de dificultad (Fácil, Medio y Difícil/Imbatible).
"""

import time
import random
from typing import Dict, List, Optional, Tuple
from model.game_model import GameModel, GameState


class MinimaxAgent:
    """Agente de Inteligencia Artificial basado en el algoritmo Minimax con Backtracking."""

    DIFFICULTY_EASY = "EASY"
    DIFFICULTY_MEDIUM = "MEDIUM"
    DIFFICULTY_HARD = "HARD"

    def __init__(
        self,
        ai_player: str = GameModel.PLAYER_O,
        human_player: str = GameModel.PLAYER_X,
        difficulty: str = DIFFICULTY_HARD,
    ) -> None:
        """Inicializa el agente Minimax.
        
        Args:
            ai_player: Símbolo con el que juega la IA ('O' o 'X').
            human_player: Símbolo con el que juega el humano ('X' u 'O').
            difficulty: Nivel de dificultad ('EASY', 'MEDIUM', 'HARD').
        """
        self.ai_player = ai_player
        self.human_player = human_player
        self.difficulty = difficulty
        self.nodes_evaluated: int = 0
        self.last_response_time_ms: float = 0.0
        self._memo: Dict[Tuple[Tuple[str, ...], bool], int] = {}

    def set_players(self, ai_player: str, human_player: str) -> None:
        """Actualiza los símbolos asignados a la IA y al humano."""
        self.ai_player = ai_player
        self.human_player = human_player
        self._memo.clear()

    def set_difficulty(self, difficulty: str) -> None:
        """Establece el nivel de dificultad de la IA."""
        self.difficulty = difficulty

    def evaluate(self, model: GameModel) -> int:
        """Función de evaluación heurística de estados terminales.
        
        Returns:
            +10 si la IA gana, -10 si el humano gana, 0 en empate o juego en curso.
        """
        winner = model.check_winner()
        if winner == self.ai_player:
            return 10
        elif winner == self.human_player:
            return -10
        return 0

    def minimax(self, model: GameModel, depth: int, is_maximizing: bool) -> int:
        """Algoritmo Minimax recursivo con Backtracking.
        
        Args:
            model: Instancia del modelo de juego que se simula.
            depth: Profundidad actual en el árbol de búsqueda.
            is_maximizing: True si es el turno del jugador maximizador (IA),
                           False si es el turno del minimizador (Humano).
                           
        Returns:
            Puntuación óptima para el subárbol evaluado.
        """
        self.nodes_evaluated += 1
        score = self.evaluate(model)

        # Casos base: victoria de IA (+10 - depth), victoria de Humano (-10 + depth)
        if score == 10:
            return score - depth
        if score == -10:
            return score + depth
        if len(model.get_available_moves()) == 0:
            return 0  # Empate

        state_key = (tuple(model.get_board()), is_maximizing)
        if state_key in self._memo:
            return self._memo[state_key]

        if is_maximizing:
            best_score = -float("inf")
            for move in model.get_available_moves():
                model.make_move(move, self.ai_player)       # Aplicar movimiento
                current_score = self.minimax(model, depth + 1, False)
                model.undo_move(move)                         # BACKTRACKING
                best_score = max(best_score, current_score)
            res = int(best_score)
        else:
            best_score = float("inf")
            for move in model.get_available_moves():
                model.make_move(move, self.human_player)    # Aplicar movimiento
                current_score = self.minimax(model, depth + 1, True)
                model.undo_move(move)                         # BACKTRACKING
                best_score = min(best_score, current_score)
            res = int(best_score)

        self._memo[state_key] = res
        return res

    def get_best_move(self, model: GameModel) -> Optional[int]:
        """Calcula el mejor movimiento para la IA según la dificultad configurada.
        
        Args:
            model: Estado actual del juego.
            
        Returns:
            Índice de la casilla elegida (0 a 8) o None si no hay movimientos disponibles.
        """
        available_moves = model.get_available_moves()
        if not available_moves:
            return None

        # Preservar el estado original del modelo para garantizar pureza y evitar efectos secundarios
        saved_current_player = model.get_current_player()
        saved_state = model.get_state()
        saved_winning_line = model.get_winning_line()

        start_time = time.perf_counter()
        self.nodes_evaluated = 0
        self._memo.clear()

        try:
            # Selección según dificultad
            if self.difficulty == self.DIFFICULTY_EASY:
                chosen_move = self._get_easy_move(model, available_moves)
            elif self.difficulty == self.DIFFICULTY_MEDIUM:
                chosen_move = self._get_medium_move(model, available_moves)
            else:  # DIFFICULTY_HARD (Imbatible)
                chosen_move = self._get_optimal_move(model, available_moves)
        finally:
            # Restaurar fielmente el estado del modelo
            model._current_player = saved_current_player
            model._state = saved_state
            model._winning_line = saved_winning_line

        elapsed = (time.perf_counter() - start_time) * 1000
        self.last_response_time_ms = round(elapsed, 2)
        return chosen_move

    def _get_optimal_move(self, model: GameModel, available_moves: List[int]) -> int:
        """Determina la jugada óptima exacta mediante Minimax (Imbatible)."""
        best_score = -float("inf")
        best_moves: List[int] = []

        for move in available_moves:
            model.make_move(move, self.ai_player)
            score = self.minimax(model, depth=0, is_maximizing=False)
            model.undo_move(move)

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return random.choice(best_moves)

    def _get_medium_move(self, model: GameModel, available_moves: List[int]) -> int:
        """Dificultad Media: Revisa victorias o bloqueos inmediatos; de lo contrario, 60% Minimax."""
        # 1. ¿Puede ganar en este turno?
        for move in available_moves:
            model.make_move(move, self.ai_player)
            is_win = (model.check_winner() == self.ai_player)
            model.undo_move(move)
            if is_win:
                return move

        # 2. ¿El oponente puede ganar en el próximo turno? ¡Bloquearlo!
        for move in available_moves:
            model.make_move(move, self.human_player)
            is_block = (model.check_winner() == self.human_player)
            model.undo_move(move)
            if is_block:
                return move

        # 3. 60% de probabilidad de jugada óptima, 40% aleatoria
        if random.random() < 0.60:
            return self._get_optimal_move(model, available_moves)
        return random.choice(available_moves)

    def _get_easy_move(self, model: GameModel, available_moves: List[int]) -> int:
        """Dificultad Fácil: 80% aleatoria, 20% jugada óptima."""
        if random.random() < 0.20:
            return self._get_optimal_move(model, available_moves)
        return random.choice(available_moves)
