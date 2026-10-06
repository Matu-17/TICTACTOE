"""Modelo del juego Tres en Raya (Tic-Tac-Toe).

Gestiona exclusivamente el estado del tablero, las reglas, validaciones,
turnos y condiciones de victoria o empate.
"""

from enum import Enum
from typing import List, Optional, Tuple


class GameState(Enum):
    """Estados posibles de la partida."""
    PLAYING = "PLAYING"
    X_WON = "X_WON"
    O_WON = "O_WON"
    DRAW = "DRAW"


class GameModel:
    """Modelo del juego Tres en Raya.
    
    Representa el tablero mediante una lista de 9 posiciones (0 a 8)
    e implementa la lógica pura del juego sin acoplamiento a interfaces gráficas.
    """

    # Combinaciones ganadoras: 3 filas, 3 columnas y 2 diagonales
    WINNING_COMBINATIONS: Tuple[Tuple[int, int, int], ...] = (
        (0, 1, 2),  # Fila 0
        (3, 4, 5),  # Fila 1
        (6, 7, 8),  # Fila 2
        (0, 3, 6),  # Columna 0
        (1, 4, 7),  # Columna 1
        (2, 5, 8),  # Columna 2
        (0, 4, 8),  # Diagonal principal
        (2, 4, 6),  # Diagonal secundaria
    )

    PLAYER_X = "X"
    PLAYER_O = "O"
    EMPTY = ""

    def __init__(self) -> None:
        """Inicializa una nueva partida."""
        self._board: List[str] = [self.EMPTY] * 9
        self._current_player: str = self.PLAYER_X
        self._state: GameState = GameState.PLAYING
        self._winning_line: Optional[Tuple[int, int, int]] = None

    def reset_game(self) -> None:
        """Reinicia el tablero y el estado de la partida."""
        self._board = [self.EMPTY] * 9
        self._current_player = self.PLAYER_X
        self._state = GameState.PLAYING
        self._winning_line = None

    def get_board(self) -> List[str]:
        """Devuelve una copia del tablero actual."""
        return list(self._board)

    def get_current_player(self) -> str:
        """Devuelve el símbolo del jugador que tiene el turno actual."""
        return self._current_player

    def get_state(self) -> GameState:
        """Devuelve el estado actual de la partida."""
        return self._state

    def get_winning_line(self) -> Optional[Tuple[int, int, int]]:
        """Devuelve las posiciones de la línea ganadora, si existe."""
        return self._winning_line

    def is_valid_move(self, position: int) -> bool:
        """Verifica si un movimiento es válido en la posición indicada."""
        if not (0 <= position <= 8):
            return False
        if self._state != GameState.PLAYING:
            return False
        return self._board[position] == self.EMPTY

    def get_available_moves(self) -> List[int]:
        """Devuelve una lista con los índices de todas las casillas disponibles."""
        return [i for i, cell in enumerate(self._board) if cell == self.EMPTY]

    def make_move(self, position: int, player: Optional[str] = None) -> bool:
        """Realiza un movimiento en el tablero.
        
        Args:
            position: Índice de la casilla (0 a 8).
            player: Jugador que realiza el movimiento (opcional, por defecto el turno actual).
            
        Returns:
            True si el movimiento se realizó con éxito, False en caso contrario.
        """
        if not self.is_valid_move(position):
            return False

        move_player = player if player is not None else self._current_player
        self._board[position] = move_player

        # Actualizar estado tras el movimiento
        self._update_game_state()

        # Si la partida continúa, alternar turno
        if self._state == GameState.PLAYING:
            self._current_player = self.PLAYER_O if self._current_player == self.PLAYER_X else self.PLAYER_X

        return True

    def undo_move(self, position: int) -> bool:
        """Deshace un movimiento en la casilla indicada (preparación para Backtracking en Semana 2).
        
        Args:
            position: Índice de la casilla (0 a 8) a liberar.
            
        Returns:
            True si se deshizo el movimiento, False si la posición era inválida o ya estaba vacía.
        """
        if not (0 <= position <= 8) or self._board[position] == self.EMPTY:
            return False

        last_player = self._board[position]
        self._board[position] = self.EMPTY
        self._current_player = last_player
        self._state = GameState.PLAYING
        self._winning_line = None
        return True

    def check_winner(self, player: Optional[str] = None) -> Optional[str]:
        """Comprueba si hay un ganador en el tablero actual.
        
        Args:
            player: Si se especifica, verifica únicamente si ese jugador ganó.
            
        Returns:
            El símbolo del ganador ('X' o 'O') o None si no hay ganador.
        """
        players_to_check = [player] if player is not None else [self.PLAYER_X, self.PLAYER_O]

        for p in players_to_check:
            for combo in self.WINNING_COMBINATIONS:
                if (self._board[combo[0]] == p and
                    self._board[combo[1]] == p and
                    self._board[combo[2]] == p):
                    return p
        return None

    def is_draw(self) -> bool:
        """Determina si la partida terminó en empate."""
        return (self.check_winner() is None) and (len(self.get_available_moves()) == 0)

    def is_game_over(self) -> bool:
        """Determina si la partida ha finalizado (victoria o empate)."""
        return self._state != GameState.PLAYING

    def _update_game_state(self) -> None:
        """Actualiza internamente el estado de la partida y registra la línea ganadora."""
        for combo in self.WINNING_COMBINATIONS:
            p0, p1, p2 = combo
            if self._board[p0] != self.EMPTY and self._board[p0] == self._board[p1] == self._board[p2]:
                winner = self._board[p0]
                self._winning_line = combo
                self._state = GameState.X_WON if winner == self.PLAYER_X else GameState.O_WON
                return

        if len(self.get_available_moves()) == 0:
            self._state = GameState.DRAW
            self._winning_line = None
        else:
            self._state = GameState.PLAYING
            self._winning_line = None
