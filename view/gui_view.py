"""Vista del juego Tres en Raya (Tic-Tac-Toe) implementada con Tkinter.

Responsabilidad exclusiva:
- Crear y posicionar los widgets gráficos de la interfaz.
- Capturar eventos del usuario (clics en casillas, botones, selector) y delegarlos al Controlador.
- Reflejar visualmente el estado del tablero y los mensajes recibidos del Controlador.

NO contiene reglas de juego ni valida condiciones de victoria o empate.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, Dict, List, Optional, Tuple


class GameView:
    """Clase principal de la interfaz gráfica del juego."""

    def __init__(self, root: tk.Tk) -> None:
        """Inicializa los componentes visuales de la ventana.
        
        Args:
            root: Ventana principal de Tkinter.
        """
        self.root = root
        self.root.title("Tres en Raya — Proyecto Educativo IA (MVC)")
        self.root.resizable(True, True)

        # Callbacks que serán registrados por el Controlador
        self._on_cell_click: Optional[Callable[[int], None]] = None
        self._on_reset_click: Optional[Callable[[], None]] = None
        self._on_mode_change: Optional[Callable[[str], None]] = None

        # Configuración de estilos y colores
        self.COLOR_BG = "#F4F6F9"
        self.COLOR_CARD = "#FFFFFF"
        self.COLOR_TEXT_MAIN = "#2C3E50"
        self.COLOR_TEXT_MUTED = "#7F8C8D"
        self.COLOR_X = "#2980B9"        # Azul
        self.COLOR_O = "#E67E22"        # Naranja
        self.COLOR_WIN = "#27AE60"      # Verde para resaltar victoria
        self.COLOR_EMPTY = "#ECF0F1"

        self.root.configure(bg=self.COLOR_BG)

        # Lista de botones del tablero 3x3
        self._buttons: List[tk.Button] = []

        # Construcción de la interfaz
        self._create_widgets()
        self._center_window(480, 720)

    def set_callbacks(
        self,
        on_cell_click: Callable[[int], None],
        on_reset_click: Callable[[], None],
        on_mode_change: Callable[[str], None],
    ) -> None:
        """Registra las funciones de callback del Controlador."""
        self._on_cell_click = on_cell_click
        self._on_reset_click = on_reset_click
        self._on_mode_change = on_mode_change

    def _create_widgets(self) -> None:
        """Crea y organiza los componentes gráficos."""
        # 1. Encabezado / Título
        header_frame = tk.Frame(self.root, bg=self.COLOR_BG)
        header_frame.pack(fill=tk.X, padx=20, pady=(8, 3))

        title_label = tk.Label(
            header_frame,
            text="TRES EN RAYA",
            font=("Segoe UI", 18, "bold"),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT_MAIN,
        )
        title_label.pack()

        subtitle_label = tk.Label(
            header_frame,
            text="Arquitectura MVC — Semana 1",
            font=("Segoe UI", 10),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT_MUTED,
        )
        subtitle_label.pack()

        # 2. Selector de Modo de Juego
        mode_frame = tk.Frame(self.root, bg=self.COLOR_BG)
        mode_frame.pack(fill=tk.X, padx=25, pady=10)

        mode_label = tk.Label(
            mode_frame,
            text="Modo de juego:",
            font=("Segoe UI", 10, "bold"),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT_MAIN,
        )
        mode_label.pack(side=tk.LEFT, padx=(0, 10))

        self.mode_var = tk.StringVar(value="Humano vs Humano")
        self.mode_selector = ttk.Combobox(
            mode_frame,
            textvariable=self.mode_var,
            state="readonly",
            font=("Segoe UI", 9),
            width=28,
            values=[
                "Humano vs Humano",
                "Humano vs Minimax",
                "Humano vs Machine Learning",
            ],
        )
        self.mode_selector.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.mode_selector.bind("<<ComboboxSelected>>", self._handle_mode_selected)

        # 3. Estado de la partida / Turno
        status_frame = tk.Frame(self.root, bg=self.COLOR_CARD, relief=tk.GROOVE, bd=1)
        status_frame.pack(fill=tk.X, padx=25, pady=5)

        self.status_label = tk.Label(
            status_frame,
            text="Turno actual: X",
            font=("Segoe UI", 12, "bold"),
            bg=self.COLOR_CARD,
            fg=self.COLOR_TEXT_MAIN,
            pady=8,
        )
        self.status_label.pack()

        # 4. Tablero 3x3
        board_container = tk.Frame(self.root, bg=self.COLOR_BG)
        board_container.pack(padx=25, pady=8)

        for i in range(9):
            row = i // 3
            col = i % 3
            btn = tk.Button(
                board_container,
                text="",
                font=("Segoe UI", 24, "bold"),
                width=4,
                height=2,
                bg=self.COLOR_EMPTY,
                activebackground="#DFE6E9",
                relief=tk.RAISED,
                bd=2,
                command=lambda idx=i: self._handle_cell_clicked(idx),
            )
            btn.grid(row=row, column=col, padx=4, pady=4)
            self._buttons.append(btn)

        # 5. Panel de Métricas / Información Preparado
        self.metrics_frame = tk.LabelFrame(
            self.root,
            text=" Panel de Información & Métricas ",
            font=("Segoe UI", 9, "bold"),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT_MAIN,
            padx=12,
            pady=6,
        )
        self.metrics_frame.pack(fill=tk.X, padx=25, pady=5)

        self.lbl_current_mode = tk.Label(
            self.metrics_frame,
            text="Modo: Humano vs Humano",
            font=("Segoe UI", 9),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT_MAIN,
            anchor="w",
        )
        self.lbl_current_mode.pack(fill=tk.X)

        self.lbl_ai_metrics = tk.Label(
            self.metrics_frame,
            text="Métricas IA: N/A",
            font=("Segoe UI", 9, "italic"),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT_MUTED,
            anchor="w",
        )
        self.lbl_ai_metrics.pack(fill=tk.X)

        # 6. Botón Nueva Partida / Reiniciar
        control_frame = tk.Frame(self.root, bg=self.COLOR_BG)
        control_frame.pack(fill=tk.X, padx=25, pady=(8, 13))

        self.btn_reset = tk.Button(
            control_frame,
            text="Nueva Partida",
            font=("Segoe UI", 11, "bold"),
            bg="#34495E",
            fg="white",
            activebackground="#2C3E50",
            activeforeground="white",
            relief=tk.FLAT,
            padx=15,
            pady=6,
            cursor="hand2",
            command=self._handle_reset_clicked,
        )
        self.btn_reset.pack(fill=tk.X)

    def _center_window(self, width: int, height: int) -> None:
        """Centra la ventana principal en la pantalla."""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2) - 40
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    # --- Manejadores internos de eventos hacia el Controlador ---

    def _handle_cell_clicked(self, index: int) -> None:
        if self._on_cell_click:
            self._on_cell_click(index)

    def _handle_reset_clicked(self) -> None:
        if self._on_reset_click:
            self._on_reset_click()

    def _handle_mode_selected(self, event=None) -> None:
        selected_mode = self.mode_var.get()
        if self._on_mode_change:
            self._on_mode_change(selected_mode)

    # --- Métodos invocados por el Controlador para actualizar la UI ---

    def update_board(
        self,
        board: List[str],
        winning_line: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Actualiza la representación visual de las 9 casillas."""
        for i in range(9):
            symbol = board[i]
            btn = self._buttons[i]
            btn.config(text=symbol)

            if symbol == "X":
                btn.config(fg=self.COLOR_X, bg=self.COLOR_EMPTY)
            elif symbol == "O":
                btn.config(fg=self.COLOR_O, bg=self.COLOR_EMPTY)
            else:
                btn.config(fg=self.COLOR_TEXT_MAIN, bg=self.COLOR_EMPTY)

        # Resaltar casillas ganadoras si aplica
        if winning_line is not None:
            for idx in winning_line:
                self._buttons[idx].config(bg=self.COLOR_WIN, fg="white")

    def update_status(self, message: str, color: Optional[str] = None) -> None:
        """Actualiza el texto del indicador de estado/turno."""
        self.status_label.config(
            text=message,
            fg=color if color else self.COLOR_TEXT_MAIN,
        )

    def update_metrics_panel(self, mode_text: str, metrics_text: str) -> None:
        """Actualiza los textos del panel de métricas."""
        self.lbl_current_mode.config(text=mode_text)
        self.lbl_ai_metrics.config(text=metrics_text)

    def set_mode_selector_value(self, value: str) -> None:
        """Establece el valor mostrado en el selector de modo."""
        self.mode_var.set(value)

    def show_info(self, title: str, message: str) -> None:
        """Muestra una ventana modal informativa."""
        messagebox.showinfo(title, message)
