"""Controlador del juego Tres en Raya

Responsabilidad:
- Orquestar el flujo de la partida.
- Responder a los eventos generados en la Vista (GameView).
- Invocar los métodos del Modelo (GameModel) para actualizar el estado del juego.
- Actualizar la Vista con el nuevo estado del Modelo tras cada acción.
- Gestionar la selección del modo de juego y preparar puntos de extensión para las Semanas 2 y 3.
"""

from typing import Optional
from model.game_model import GameModel, GameState
from view.gui_view import GameView


class GameController:
    """Controlador que vincula la lógica del Modelo con la interfaz de la Vista."""

    MODE_HUMAN_VS_HUMAN = "Humano vs Humano"
    MODE_HUMAN_VS_MINIMAX = "Humano vs Minimax"
    MODE_HUMAN_VS_ML = "Humano vs Machine Learning"

    def __init__(self, model: GameModel, view: GameView) -> None:
        """Inicializa el controlador registrando los callbacks de la vista.
        
        Args:
            model: Instancia del modelo de juego.
            view: Instancia de la vista gráfica.
        """
        self.model = model
        self.view = view
        self.current_mode: str = self.MODE_HUMAN_VS_HUMAN

        # Conectar callbacks de la Vista hacia los métodos del Controlador
        self.view.set_callbacks(
            on_cell_click=self.handle_cell_click,
            on_reset_click=self.handle_reset_game,
            on_mode_change=self.handle_mode_change,
        )

        # Actualizar la vista al estado inicial
        self._refresh_view()

    def handle_cell_click(self, position: int) -> None:
        """Maneja el evento de pulsación de una casilla del tablero.
        
        Args:
            position: Índice de la casilla pulsada (0 a 8).
        """
        # Si la partida ya finalizó, ignorar clics
        if self.model.is_game_over():
            return

        # Intentar realizar el movimiento a través del modelo
        move_success = self.model.make_move(position)
        if not move_success:
            # Movimiento inválido (casilla ocupada o partida terminada)
            return

        # Actualizar la vista tras el movimiento
        self._refresh_view()

    def handle_reset_game(self) -> None:
        """Maneja el evento de reinicio de la partida."""
        self.model.reset_game()
        self._refresh_view()

    def handle_mode_change(self, selected_mode: str) -> None:
        """Maneja la selección de un nuevo modo de juego.
        
        Args:
            selected_mode: Cadena con el nombre del modo seleccionado.
        """
        if selected_mode == self.MODE_HUMAN_VS_HUMAN:
            self.current_mode = self.MODE_HUMAN_VS_HUMAN
            self.handle_reset_game()
        elif selected_mode == self.MODE_HUMAN_VS_MINIMAX:
            self.view.show_info(
                "Modo no disponible",
                "El modo 'Humano vs Minimax' (Backtracking y búsqueda de árbol) "
                "será implementado durante la Semana 2 del proyecto.",
            )
            # Restaurar selector al modo activo actual
            self.view.set_mode_selector_value(self.current_mode)
        elif selected_mode == self.MODE_HUMAN_VS_ML:
            self.view.show_info(
                "Modo no disponible",
                "El modo 'Humano vs Machine Learning' (DecisionTreeClassifier con dataset sintético) "
                "será implementado durante la Semana 3 del proyecto.",
            )
            # Restaurar selector al modo activo actual
            self.view.set_mode_selector_value(self.current_mode)

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
            self.view.update_status(f"Turno actual: Jugador {current_player}")
        elif state == GameState.X_WON:
            self.view.update_status("🎉 ¡Victoria del Jugador X!", color="#27AE60")
        elif state == GameState.O_WON:
            self.view.update_status("🎉 ¡Victoria del Jugador O!", color="#27AE60")
        elif state == GameState.DRAW:
            self.view.update_status("🤝 ¡Partida terminada en Empate!", color="#7F8C8D")

        # 3. Actualizar panel de información y métricas
        self.view.update_metrics_panel(
            mode_text=f"Modo: {self.current_mode}",
            metrics_text="Métricas IA: N/A",
        )
