"""Punto de entrada principal del juego Tres en Raya (Tic-Tac-Toe).

Inicializa el Modelo, la Vista (Tkinter) y el Controlador según el patrón MVC.
"""

import tkinter as tk
from model.game_model import GameModel
from view.gui_view import GameView
from controller.game_controller import GameController


def main() -> None:
    """Función principal de ejecución."""
    root = tk.Tk()
    
    model = GameModel()
    view = GameView(root)
    controller = GameController(model=model, view=view)
    
    root.mainloop()


if __name__ == "__main__":
    main()
