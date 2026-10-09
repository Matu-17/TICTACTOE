"""Controlador del juego Tres en Raya (Tic-Tac-Toe).

Responsabilidad:
- Orquestar el flujo de la partida en todos los modos (Humano vs Humano, Minimax y Machine Learning).
- Responder a los eventos generados en la Vista (GameView).
- Invocar los métodos del Modelo (GameModel) para actualizar el estado del juego.
- Coordinar las decisiones de los agentes inteligentes (MinimaxAgent y MLAgent).
- Medir y publicar métricas de rendimiento (nodos explorados, tiempo de respuesta y precisión).
"""

import os
from typing import Optional
from model.game_model import GameModel, GameState
from model.minimax_agent import MinimaxAgent
from model.ml_agent import MLAgent
from view.gui_view import GameView


class GameController:
    """Controlador que vincula la lógica del Modelo con la interfaz de la Vista y los Agentes IA."""

    MODE_HUMAN_VS_HUMAN = "Humano vs Humano"
    MODE_HUMAN_VS_MINIMAX = "Humano vs Minimax (Semana 2)"
    MODE_HUMAN_VS_ML = "Humano vs Machine Learning (Semana 3)"

    def __init__(self, model: GameModel, view: GameView) -> None:
        """Inicializa el controlador registrando los callbacks de la vista.
        
        Args:
            model: Instancia del modelo de juego.
            view: Instancia de la vista gráfica.
        """
        self.model = model
        self.view = view
        self.current_mode: str = self.MODE_HUMAN_VS_HUMAN

        # Símbolos asignados
        self.human_symbol: str = "X"
        self.ai_symbol: str = "O"

        # Inicializar agentes de IA
        self.minimax_agent = MinimaxAgent(
            ai_player=self.ai_symbol,
            human_player=self.human_symbol,
            difficulty=MinimaxAgent.DIFFICULTY_HARD,
        )
        self.ml_agent = MLAgent(
            ai_player=self.ai_symbol,
            dataset_path="data/dataset.csv",
        )

        # Conectar callbacks de la Vista hacia los métodos del Controlador
        self.view.set_callbacks(
            on_cell_click=self.handle_cell_click,
            on_reset_click=self.handle_reset_game,
            on_mode_change=self.handle_mode_change,
            on_symbol_change=self.handle_symbol_change,
            on_difficulty_change=self.handle_difficulty_change,
            on_train_dataset_click=self.handle_train_dataset,
            on_view_tree_click=self.handle_view_tree,
        )

        # Cargar y entrenar el agente ML en segundo plano si aún no está entrenado
        self._ensure_ml_agent_ready()

        # Actualizar la vista al estado inicial
        self._refresh_view()

    def _ensure_ml_agent_ready(self) -> None:
        """Asegura que el modelo ML esté entrenado y disponible."""
        if not self.ml_agent.is_trained:
            self.ml_agent.train()

    def handle_cell_click(self, position: int) -> None:
        """Maneja el evento de pulsación de una casilla del tablero."""
        if self.model.is_game_over():
            return

        # Si estamos en modo IA y no es el turno del humano, ignorar clic
        current_turn = self.model.get_current_player()
        if self.current_mode != self.MODE_HUMAN_VS_HUMAN and current_turn != self.human_symbol:
            return

        # Realizar el movimiento del humano en el modelo
        move_success = self.model.make_move(position)
        if not move_success:
            return

        self._refresh_view()

        # Si el juego continúa y es modo contra IA, programar jugada de la IA
        if not self.model.is_game_over() and self.current_mode != self.MODE_HUMAN_VS_HUMAN:
            # Pequeña pausa (300 ms) para una experiencia de usuario natural
            self.view.root.after(300, self._execute_ai_turn)

    def _execute_ai_turn(self) -> None:
        """Ejecuta el turno de la IA según el modo de juego seleccionado."""
        if self.model.is_game_over():
            return

        ai_move: Optional[int] = None

        if self.current_mode == self.MODE_HUMAN_VS_MINIMAX:
            ai_move = self.minimax_agent.get_best_move(self.model)
            nodes = self.minimax_agent.nodes_evaluated
            latency = self.minimax_agent.last_response_time_ms
            self.view.update_metrics(nodes=nodes, latency_ms=latency)

        elif self.current_mode == self.MODE_HUMAN_VS_ML:
            ai_move = self.ml_agent.get_best_move(self.model)
            latency = self.ml_agent.last_response_time_ms
            accuracy = self.ml_agent.tree.accuracy
            self.view.update_metrics(nodes=0, latency_ms=latency, ml_accuracy=accuracy)

        if ai_move is not None:
            self.model.make_move(ai_move, self.ai_symbol)
            self._refresh_view()

    def handle_reset_game(self) -> None:
        """Maneja el evento de reinicio de la partida."""
        self.model.reset_game()
        self._refresh_view()

        # Si el humano eligió 'O' y juega contra IA, la IA debe mover primero
        if self.current_mode != self.MODE_HUMAN_VS_HUMAN and self.human_symbol == "O":
            self.view.root.after(350, self._execute_ai_turn)

    def handle_mode_change(self, selected_mode: str) -> None:
        """Maneja la selección de un nuevo modo de juego."""
        self.current_mode = selected_mode
        self._update_agent_symbols()
        self.handle_reset_game()

    def handle_symbol_change(self, chosen_symbol: str) -> None:
        """Maneja el cambio de símbolo elegido por el usuario (X u O)."""
        self.human_symbol = chosen_symbol
        self.ai_symbol = "O" if chosen_symbol == "X" else "X"
        self._update_agent_symbols()
        self.handle_reset_game()

    def handle_difficulty_change(self, difficulty_text: str) -> None:
        """Maneja el cambio de dificultad para el agente Minimax."""
        if "Fácil" in difficulty_text:
            self.minimax_agent.set_difficulty(MinimaxAgent.DIFFICULTY_EASY)
        elif "Medio" in difficulty_text:
            self.minimax_agent.set_difficulty(MinimaxAgent.DIFFICULTY_MEDIUM)
        else:
            self.minimax_agent.set_difficulty(MinimaxAgent.DIFFICULTY_HARD)

    def handle_train_dataset(self) -> None:
        """Genera un nuevo dataset de 1000 partidas y reentrena el Árbol Binario."""
        self.ml_agent.generate_dataset(num_games=1000)
        self.ml_agent.train()
        self.view.show_info(
            "Entrenamiento Completado",
            f"Dataset de 1000+ partidas generado con éxito.\nPrecisión del Árbol Binario: {self.ml_agent.tree.accuracy:.2f}%",
        )
        self._refresh_view()

    def handle_view_tree(self) -> None:
        """Abre la ventana para visualizar el Árbol Binario de Decisión."""
        self._ensure_ml_agent_ready()
        rules_text = self.ml_agent.tree.export_text(max_lines=60)
        image_path = os.path.abspath("data/decision_tree.png")
        if not os.path.exists(image_path):
            self.ml_agent.tree.export_plot(image_path)
        self.view.show_tree_window(rules_text, image_path)

    def _update_agent_symbols(self) -> None:
        """Sincroniza los símbolos configurados con los agentes de IA."""
        self.minimax_agent.set_players(ai_player=self.ai_symbol, human_player=self.human_symbol)
        self.ml_agent.set_player(ai_player=self.ai_symbol)

    def _refresh_view(self) -> None:
        """Sincroniza la Vista con el estado actual del Modelo."""
        board = self.model.get_board()
        state = self.model.get_state()
        winning_line = self.model.get_winning_line()

        # 1. Actualizar tablero
        self.view.update_board(board, winning_line)

        # 2. Actualizar texto de estado / turno
        if state == GameState.PLAYING:
            current_player = self.model.get_current_player()
            if self.current_mode == self.MODE_HUMAN_VS_HUMAN:
                self.view.update_status(f"Turno actual: Jugador {current_player}")
            else:
                is_human_turn = (current_player == self.human_symbol)
                agent_name = "Minimax" if self.current_mode == self.MODE_HUMAN_VS_MINIMAX else "Árbol ML"
                turn_msg = f"Tu turno ({current_player})" if is_human_turn else f"Pensando IA {agent_name} ({current_player})..."
                self.view.update_status(turn_msg)
        elif state == GameState.X_WON:
            winner_text = "🎉 ¡Victoria del Jugador X!" if self.current_mode == self.MODE_HUMAN_VS_HUMAN or self.human_symbol == "X" else "🤖 ¡Victoria de la IA (X)!"
            self.view.update_status(winner_text, color="#27AE60")
        elif state == GameState.O_WON:
            winner_text = "🎉 ¡Victoria del Jugador O!" if self.current_mode == self.MODE_HUMAN_VS_HUMAN or self.human_symbol == "O" else "🤖 ¡Victoria de la IA (O)!"
            self.view.update_status(winner_text, color="#27AE60")
        elif state == GameState.DRAW:
            self.view.update_status("🤝 ¡Partida terminada en Empate!", color="#7F8C8D")
