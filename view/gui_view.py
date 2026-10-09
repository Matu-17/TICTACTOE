"""Vista del juego Tres en Raya (Tic-Tac-Toe) implementada con Tkinter.

Responsabilidad exclusiva:
- Crear y posicionar los widgets gráficos de la interfaz.
- Capturar eventos del usuario (clics en casillas, botones, selector) y delegarlos al Controlador.
- Reflejar visualmente el estado del tablero, métricas y mensajes recibidos del Controlador.

NO contiene reglas de juego ni valida condiciones de victoria o empate.
"""

import os
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
        self.root.resizable(False, False)

        # Callbacks registrados por el Controlador
        self._on_cell_click: Optional[Callable[[int], None]] = None
        self._on_reset_click: Optional[Callable[[], None]] = None
        self._on_mode_change: Optional[Callable[[str], None]] = None
        self._on_symbol_change: Optional[Callable[[str], None]] = None
        self._on_difficulty_change: Optional[Callable[[str], None]] = None
        self._on_train_dataset_click: Optional[Callable[[], None]] = None
        self._on_view_tree_click: Optional[Callable[[], None]] = None

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
        self._center_window(520, 720)

    def set_callbacks(
        self,
        on_cell_click: Callable[[int], None],
        on_reset_click: Callable[[], None],
        on_mode_change: Callable[[str], None],
        on_symbol_change: Callable[[str], None],
        on_difficulty_change: Callable[[str], None],
        on_train_dataset_click: Callable[[], None],
        on_view_tree_click: Callable[[], None],
    ) -> None:
        """Registra las funciones de callback del Controlador."""
        self._on_cell_click = on_cell_click
        self._on_reset_click = on_reset_click
        self._on_mode_change = on_mode_change
        self._on_symbol_change = on_symbol_change
        self._on_difficulty_change = on_difficulty_change
        self._on_train_dataset_click = on_train_dataset_click
        self._on_view_tree_click = on_view_tree_click

    def _create_widgets(self) -> None:
        """Crea y organiza los componentes gráficos."""
        # 1. Encabezado / Título
        header_frame = tk.Frame(self.root, bg=self.COLOR_BG)
        header_frame.pack(fill=tk.X, padx=20, pady=(10, 5))

        title_label = tk.Label(
            header_frame,
            text="TRES EN RAYA — IA & ML",
            font=("Segoe UI", 16, "bold"),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT_MAIN,
        )
        title_label.pack()

        subtitle_label = tk.Label(
            header_frame,
            text="Arquitectura MVC • Minimax Backtracking • Árbol de Decisión",
            font=("Segoe UI", 9),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT_MUTED,
        )
        subtitle_label.pack()

        # 2. Panel de Configuración (Modo, Símbolo y Dificultad)
        config_frame = tk.LabelFrame(
            self.root,
            text=" Configuración de Partida ",
            font=("Segoe UI", 9, "bold"),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT_MAIN,
            padx=10,
            pady=6,
        )
        config_frame.pack(fill=tk.X, padx=20, pady=5)

        # Fila 1: Selector de Modo
        row1 = tk.Frame(config_frame, bg=self.COLOR_BG)
        row1.pack(fill=tk.X, pady=2)
        tk.Label(row1, text="Modo:", font=("Segoe UI", 9, "bold"), bg=self.COLOR_BG, fg=self.COLOR_TEXT_MAIN).pack(side=tk.LEFT, padx=(0, 5))
        self.mode_var = tk.StringVar(value="Humano vs Humano")
        self.mode_selector = ttk.Combobox(
            row1,
            textvariable=self.mode_var,
            state="readonly",
            font=("Segoe UI", 9),
            values=[
                "Humano vs Humano",
                "Humano vs Minimax (Semana 2)",
                "Humano vs Machine Learning (Semana 3)",
            ],
        )
        self.mode_selector.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.mode_selector.bind("<<ComboboxSelected>>", self._handle_mode_selected)

        # Fila 2: Selector de Símbolo y Dificultad
        self.row2 = tk.Frame(config_frame, bg=self.COLOR_BG)
        self.row2.pack(fill=tk.X, pady=4)

        tk.Label(self.row2, text="Juegas como:", font=("Segoe UI", 9), bg=self.COLOR_BG, fg=self.COLOR_TEXT_MAIN).pack(side=tk.LEFT, padx=(0, 4))
        self.symbol_var = tk.StringVar(value="X (Primero)")
        self.symbol_selector = ttk.Combobox(
            self.row2,
            textvariable=self.symbol_var,
            state="readonly",
            width=12,
            font=("Segoe UI", 9),
            values=["X (Primero)", "O (Segundo)"],
        )
        self.symbol_selector.pack(side=tk.LEFT, padx=(0, 10))
        self.symbol_selector.bind("<<ComboboxSelected>>", self._handle_symbol_selected)

        self.lbl_difficulty = tk.Label(self.row2, text="Dificultad:", font=("Segoe UI", 9), bg=self.COLOR_BG, fg=self.COLOR_TEXT_MAIN)
        self.lbl_difficulty.pack(side=tk.LEFT, padx=(0, 4))
        self.diff_var = tk.StringVar(value="Difícil (Imbatible)")
        self.diff_selector = ttk.Combobox(
            self.row2,
            textvariable=self.diff_var,
            state="readonly",
            width=16,
            font=("Segoe UI", 9),
            values=["Difícil (Imbatible)", "Medio", "Fácil"],
        )
        self.diff_selector.pack(side=tk.LEFT)
        self.diff_selector.bind("<<ComboboxSelected>>", self._handle_difficulty_selected)

        # 3. Estado de la partida / Turno
        status_frame = tk.Frame(self.root, bg=self.COLOR_CARD, relief=tk.GROOVE, bd=1)
        status_frame.pack(fill=tk.X, padx=20, pady=5)

        self.status_label = tk.Label(
            status_frame,
            text="Turno actual: Jugador X",
            font=("Segoe UI", 11, "bold"),
            bg=self.COLOR_CARD,
            fg=self.COLOR_TEXT_MAIN,
            pady=6,
        )
        self.status_label.pack()

        # 4. Tablero 3x3
        board_container = tk.Frame(self.root, bg=self.COLOR_BG)
        board_container.pack(padx=20, pady=8)

        for i in range(9):
            row = i // 3
            col = i % 3
            btn = tk.Button(
                board_container,
                text="",
                font=("Segoe UI", 22, "bold"),
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

        # 5. Panel de Métricas / Información en Tiempo Real
        self.metrics_frame = tk.LabelFrame(
            self.root,
            text=" Panel de Métricas IA en Tiempo Real ",
            font=("Segoe UI", 9, "bold"),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT_MAIN,
            padx=10,
            pady=4,
        )
        self.metrics_frame.pack(fill=tk.X, padx=20, pady=4)

        self.lbl_metric_nodes = tk.Label(
            self.metrics_frame,
            text="Nodos explorados (Minimax): 0",
            font=("Segoe UI", 8),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT_MAIN,
            anchor="w",
        )
        self.lbl_metric_nodes.pack(fill=tk.X)

        self.lbl_metric_time = tk.Label(
            self.metrics_frame,
            text="Latencia de decisión: 0.00 ms",
            font=("Segoe UI", 8),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT_MAIN,
            anchor="w",
        )
        self.lbl_metric_time.pack(fill=tk.X)

        self.lbl_metric_ml = tk.Label(
            self.metrics_frame,
            text="Precisión Árbol Binario: Listo (100% en dataset)",
            font=("Segoe UI", 8),
            bg=self.COLOR_BG,
            fg=self.COLOR_TEXT_MUTED,
            anchor="w",
        )
        self.lbl_metric_ml.pack(fill=tk.X)

        # 6. Botones de Control
        control_frame = tk.Frame(self.root, bg=self.COLOR_BG)
        control_frame.pack(fill=tk.X, padx=20, pady=(6, 10))

        self.btn_reset = tk.Button(
            control_frame,
            text="🔄 Nueva Partida",
            font=("Segoe UI", 10, "bold"),
            bg="#34495E",
            fg="white",
            activebackground="#2C3E50",
            activeforeground="white",
            relief=tk.FLAT,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._handle_reset_clicked,
        )
        self.btn_reset.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_view_tree = tk.Button(
            control_frame,
            text="🌳 Ver Árbol ML",
            font=("Segoe UI", 10, "bold"),
            bg="#27AE60",
            fg="white",
            activebackground="#219653",
            activeforeground="white",
            relief=tk.FLAT,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._handle_view_tree_clicked,
        )
        self.btn_view_tree.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

    def _center_window(self, width: int, height: int) -> None:
        """Centra la ventana principal en la pantalla."""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
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

    def _handle_symbol_selected(self, event=None) -> None:
        symbol = "X" if "X" in self.symbol_var.get() else "O"
        if self._on_symbol_change:
            self._on_symbol_change(symbol)

    def _handle_difficulty_selected(self, event=None) -> None:
        diff_text = self.diff_var.get()
        if self._on_difficulty_change:
            self._on_difficulty_change(diff_text)

    def _handle_view_tree_clicked(self) -> None:
        if self._on_view_tree_click:
            self._on_view_tree_click()

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

    def update_metrics(
        self,
        nodes: Optional[int] = None,
        latency_ms: Optional[float] = None,
        ml_accuracy: Optional[float] = None,
        custom_note: Optional[str] = None,
    ) -> None:
        """Actualiza los valores del panel de métricas."""
        if nodes is not None:
            self.lbl_metric_nodes.config(text=f"Nodos explorados (Minimax): {nodes:,}")
        else:
            self.lbl_metric_nodes.config(text="Nodos explorados (Minimax): N/A")

        if latency_ms is not None:
            self.lbl_metric_time.config(text=f"Latencia de decisión: {latency_ms:.2f} ms")
        else:
            self.lbl_metric_time.config(text="Latencia de decisión: 0.00 ms")

        if ml_accuracy is not None:
            self.lbl_metric_ml.config(text=f"Precisión Árbol Binario: {ml_accuracy:.2f}% ({custom_note or 'Dataset 1000+ partidas'})")
        elif custom_note:
            self.lbl_metric_ml.config(text=custom_note)

    def set_mode_selector_value(self, value: str) -> None:
        """Establece el valor mostrado en el selector de modo."""
        self.mode_var.set(value)

    def show_info(self, title: str, message: str) -> None:
        """Muestra una ventana modal informativa."""
        messagebox.showinfo(title, message)

    def show_tree_window(self, rules_text: str, image_path: str) -> None:
        """Abre una ventana secundaria para visualizar el Árbol Binario de Decisión."""
        tree_window = tk.Toplevel(self.root)
        tree_window.title("Visualizador de Árbol Binario de Decisión")
        tree_window.geometry("850x650")
        tree_window.configure(bg=self.COLOR_BG)

        notebook = ttk.Notebook(tree_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Pestaña 1: Gráfico con Matplotlib
        tab_graph = tk.Frame(notebook, bg=self.COLOR_BG)
        notebook.add(tab_graph, text="📊 Diagrama del Árbol")

        if os.path.exists(image_path):
            try:
                from PIL import Image, ImageTk
                pil_img = Image.open(image_path)
                pil_img.thumbnail((800, 560), Image.Resampling.LANCZOS)
                tk_img = ImageTk.PhotoImage(pil_img)

                img_label = tk.Label(tab_graph, image=tk_img, bg=self.COLOR_BG)
                img_label.image = tk_img  # Mantener referencia
                img_label.pack(padx=10, pady=10, expand=True)
            except ImportError:
                tk.Label(
                    tab_graph,
                    text="El diagrama PNG está guardado en data/decision_tree.png.\n"
                         "Para visualizarlo dentro de Tkinter instale 'pillow': py -m pip install pillow\n"
                         "Consulte las reglas lógicas en la siguiente pestaña.",
                    font=("Segoe UI", 10),
                    bg=self.COLOR_BG,
                    fg=self.COLOR_TEXT_MAIN,
                    pady=30,
                ).pack(expand=True)
            except Exception as e:
                tk.Label(tab_graph, text=f"Error cargando imagen: {e}").pack(pady=20)
        else:
            tk.Label(tab_graph, text="El gráfico aún no ha sido generado. Entrene el modelo primero.").pack(pady=20)


        # Pestaña 2: Reglas Lógicas en Texto (export_text)
        tab_rules = tk.Frame(notebook, bg=self.COLOR_BG)
        notebook.add(tab_rules, text="📜 Reglas Lógicas (IF-THEN)")

        txt_box = tk.Text(tab_rules, font=("Consolas", 10), wrap=tk.NONE, bg="#2C3E50", fg="#ECF0F1")
        v_scroll = tk.Scrollbar(tab_rules, orient=tk.VERTICAL, command=txt_box.yview)
        h_scroll = tk.Scrollbar(tab_rules, orient=tk.HORIZONTAL, command=txt_box.xview)
        txt_box.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        txt_box.insert(tk.END, rules_text)
        txt_box.config(state=tk.DISABLED)

        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        txt_box.pack(fill=tk.BOTH, expand=True)
