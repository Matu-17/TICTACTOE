"""Módulo del Agente de Machine Learning con Árbol Binario de Decisión (Semana 3).

Implementa desde cero (sin bibliotecas de caja negra como scikit-learn):
1. Un Árbol de Decisión Binario (BinaryDecisionTree) basado en impureza de Gini.
2. Generador de dataset sintético (1000+ partidas simuladas entre Minimax y jugadores aleatorios).
3. Exportador de reglas en formato texto (export_text).
4. Exportador visual del árbol en formato gráfico con Matplotlib (export_plot).
5. Inferencia en tiempo real del mejor movimiento y cálculo de métricas (latencia y precisión).
"""

import csv
import os
import random
import time
from typing import Dict, List, Optional, Tuple

from model.game_model import GameModel
from model.minimax_agent import MinimaxAgent



class TreeNode:
    """Nodo para la estructura del Árbol Binario de Decisión."""

    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None,
        value: Optional[int] = None,
        is_leaf: bool = False,
        samples: int = 0,
        impurity: float = 0.0,
        class_distribution: Optional[Dict[int, int]] = None,
    ) -> None:
        self.feature = feature                  # Índice de la casilla (0 a 8)
        self.threshold = threshold              # Umbral de división
        self.left = left                        # Rama izquierda (<= threshold)
        self.right = right                      # Rama derecha (> threshold)
        self.value = value                      # Jugada recomendada en hoja
        self.is_leaf = is_leaf                  # Es nodo terminal
        self.samples = samples                  # Cantidad de muestras
        self.impurity = impurity                # Impureza de Gini
        self.class_distribution = class_distribution or {}


class BinaryDecisionTree:
    """Clasificador de Árbol de Decisión Binario implementado desde cero."""

    def __init__(self, max_depth: int = 12, min_samples_split: int = 2) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Optional[TreeNode] = None
        self.accuracy: float = 0.0

    @staticmethod
    def _gini_impurity(y: List[int]) -> float:
        """Calcula el índice de impureza de Gini para una lista de etiquetas."""
        if not y:
            return 0.0
        total = len(y)
        counts: Dict[int, int] = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        return 1.0 - sum((count / total) ** 2 for count in counts.values())

    @staticmethod
    def _majority_class(y: List[int]) -> int:
        """Determina la clase más frecuente."""
        counts: Dict[int, int] = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        return max(counts.items(), key=lambda item: item[1])[0]

    def _best_split(
        self, X: List[List[float]], y: List[int]
    ) -> Tuple[Optional[int], Optional[float], float]:
        """Encuentra la mejor característica y umbral para dividir los datos."""
        best_gain = -1.0
        best_feature = None
        best_threshold = None
        n_samples = len(y)
        current_impurity = self._gini_impurity(y)

        if n_samples < self.min_samples_split or current_impurity == 0.0:
            return None, None, 0.0

        n_features = len(X[0])
        # Umbrales candidatos para estados de casilla (-1, 0, 1): -0.5 y 0.5
        threshold_candidates = [-0.5, 0.5]

        for feature in range(n_features):
            for threshold in threshold_candidates:
                left_y = [y[i] for i in range(n_samples) if X[i][feature] <= threshold]
                right_y = [y[i] for i in range(n_samples) if X[i][feature] > threshold]

                if not left_y or not right_y:
                    continue

                p_left = len(left_y) / n_samples
                p_right = len(right_y) / n_samples
                impurity_after = p_left * self._gini_impurity(left_y) + p_right * self._gini_impurity(right_y)
                gain = current_impurity - impurity_after

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X: List[List[float]], y: List[int], depth: int = 0) -> TreeNode:
        """Construye recursivamente el árbol binario de decisión."""
        n_samples = len(y)
        current_impurity = self._gini_impurity(y)
        counts: Dict[int, int] = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1

        majority = self._majority_class(y)

        # Condición de parada (hoja)
        if depth >= self.max_depth or n_samples < self.min_samples_split or current_impurity == 0.0:
            return TreeNode(
                value=majority,
                is_leaf=True,
                samples=n_samples,
                impurity=current_impurity,
                class_distribution=counts,
            )

        best_feature, best_threshold, best_gain = self._best_split(X, y)

        if best_feature is None or best_gain <= 0.0:
            return TreeNode(
                value=majority,
                is_leaf=True,
                samples=n_samples,
                impurity=current_impurity,
                class_distribution=counts,
            )

        # Dividir datos
        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(n_samples):
            if X[i][best_feature] <= best_threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])

        left_child = self._build_tree(left_X, left_y, depth + 1)
        right_child = self._build_tree(right_X, right_y, depth + 1)

        return TreeNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            value=majority,
            is_leaf=False,
            samples=n_samples,
            impurity=current_impurity,
            class_distribution=counts,
        )

    def fit(self, X: List[List[float]], y: List[int]) -> None:
        """Entrena el árbol binario con los datos X y etiquetas y."""
        self.root = self._build_tree(X, y, depth=0)
        # Calcular precisión en el dataset de entrenamiento
        predictions = self.predict(X)
        correct = sum(1 for p, actual in zip(predictions, y) if p == actual)
        self.accuracy = round((correct / len(y)) * 100, 2) if y else 0.0

    def predict_row(self, x: List[float], node: Optional[TreeNode] = None) -> int:
        """Predice la jugada recomendada para un solo estado de tablero."""
        if node is None:
            node = self.root
        if node is None or node.is_leaf or node.feature is None or node.threshold is None:
            return node.value if node and node.value is not None else 0

        if x[node.feature] <= node.threshold:
            return self.predict_row(x, node.left)
        else:
            return self.predict_row(x, node.right)

    def predict(self, X: List[List[float]]) -> List[int]:
        """Predice las mejores jugadas para un lote de estados de tablero."""
        return [self.predict_row(row) for row in X]

    def export_text(self, node: Optional[TreeNode] = None, depth: int = 0, max_lines: int = 30) -> str:
        """Exporta las reglas de decisión en formato texto indentado (IF-THEN)."""
        if node is None:
            node = self.root
        if node is None:
            return "Árbol no entrenado."

        lines: List[str] = []

        def _recurse(n: TreeNode, d: int) -> None:
            if len(lines) >= max_lines:
                return
            indent = "  " * d
            if n.is_leaf:
                lines.append(f"{indent}└── [HOJA] Jugada óptima = Casilla {n.value} (muestras={n.samples}, gini={n.impurity:.3f})")
            else:
                condition_desc = "vacía/O" if n.threshold == -0.5 else "vacía/X"
                lines.append(f"{indent}├── IF Casilla[{n.feature}] <= {n.threshold:.1f} ({condition_desc}):")
                if n.left:
                    _recurse(n.left, d + 1)
                lines.append(f"{indent}└── ELSE (Casilla[{n.feature}] > {n.threshold:.1f}):")
                if n.right:
                    _recurse(n.right, d + 1)

        _recurse(node, depth)
        if len(lines) >= max_lines:
            lines.append("  ... [reglas adicionales omitidas para brevedad]")
        return "\n".join(lines)

    def export_plot(self, save_path: str = "data/decision_tree.png") -> str:
        """Genera un diagrama gráfico del árbol de decisión con Matplotlib si está disponible."""
        if self.root is None:
            return ""

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return ""

        try:
            fig, ax = plt.subplots(figsize=(14, 8), dpi=120)
            ax.set_title("Árbol Binario de Decisión — Tres en Raya (Aprendizaje Supervisado)", fontsize=14, fontweight="bold", pad=15)
            ax.axis("off")

            # Dibujar recursivamente los nodos
            def _draw_node(n: TreeNode, x: float, y: float, dx: float, dy: float, depth: int) -> None:
                if depth > 4:  # Limitar visualización gráfica a los primeros 4 niveles para claridad
                    ax.text(x, y, "...", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", fc="#BDC3C7", ec="none"))
                    return

                if n.is_leaf:
                    box_text = f"Hoja\nCasilla {n.value}\n(n={n.samples})"
                    box_color = "#2ECC71"  # Verde
                else:
                    box_text = f"Casilla[{n.feature}] <= {n.threshold:.1f}\nGini: {n.impurity:.2f}\n(n={n.samples})"
                    box_color = "#3498DB"  # Azul

                ax.text(
                    x, y, box_text,
                    ha="center", va="center", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.5", fc=box_color, ec="#2C3E50", alpha=0.9, lw=1.2),
                    color="white", fontweight="bold"
                )

                if not n.is_leaf and n.left and n.right:
                    # Rama izquierda
                    x_left = x - dx
                    y_child = y - dy
                    ax.annotate(
                        "Sí", xy=(x, y - 0.04), xytext=(x_left, y_child + 0.04),
                        arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.5),
                        fontsize=8, color="#27AE60", fontweight="bold", ha="center"
                    )
                    _draw_node(n.left, x_left, y_child, dx * 0.52, dy, depth + 1)

                    # Rama derecha
                    x_right = x + dx
                    ax.annotate(
                        "No", xy=(x, y - 0.04), xytext=(x_right, y_child + 0.04),
                        arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.5),
                        fontsize=8, color="#E74C3C", fontweight="bold", ha="center"
                    )
                    _draw_node(n.right, x_right, y_child, dx * 0.52, dy, depth + 1)

            _draw_node(self.root, 0.5, 0.92, 0.24, 0.18, 0)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            return save_path
        except Exception:
            return ""



class MLAgent:
    """Agente de Inteligencia Artificial basado en el Árbol Binario de Decisión."""

    def __init__(
        self,
        ai_player: str = GameModel.PLAYER_O,
        dataset_path: str = "data/dataset.csv",
    ) -> None:
        self.ai_player = ai_player
        self.dataset_path = dataset_path
        self.tree = BinaryDecisionTree(max_depth=12, min_samples_split=2)
        self.is_trained = False
        self.last_response_time_ms: float = 0.0

    def set_player(self, ai_player: str) -> None:
        """Establece el símbolo de la IA."""
        self.ai_player = ai_player

    @staticmethod
    def encode_board(board: List[str]) -> List[float]:
        """Codifica el tablero a valores numéricos: 'X' -> 1.0, 'O' -> -1.0, '' -> 0.0."""
        mapping = {"X": 1.0, "O": -1.0, "": 0.0}
        return [mapping[cell] for cell in board]

    def generate_dataset(self, num_games: int = 1000) -> str:
        """Simula partidas automáticas y genera el archivo data/dataset.csv.
        
        Las partidas combinan simulaciones de Minimax y jugadores aleatorios para
        descubrir estados de juego diversos y mapear cada tablero a la mejor jugada.
        """
        print(f"\n[ML Agent] Dataset no encontrado o solicitud de regeneración.")
        print(f"[ML Agent] Iniciando simulación de {num_games:,} partidas automáticas...")
        start_gen_time = time.perf_counter()

        os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)
        dataset_records: List[Tuple[List[float], int]] = []
        seen_states = set()

        expert_x = MinimaxAgent(ai_player="X", human_player="O")
        expert_o = MinimaxAgent(ai_player="O", human_player="X")

        step = max(1, num_games // 4)
        for g in range(1, num_games + 1):
            sim_model = GameModel()
            
            while not sim_model.is_game_over():
                current_player = sim_model.get_current_player()
                board = sim_model.get_board()
                encoded_board = self.encode_board(board)
                state_key = tuple(encoded_board)

                # Calcular mejor jugada de Minimax para el jugador en turno
                expert = expert_x if current_player == "X" else expert_o
                best_move = expert._get_optimal_move(sim_model, sim_model.get_available_moves())

                if state_key not in seen_states and best_move is not None:
                    seen_states.add(state_key)
                    dataset_records.append((encoded_board, best_move))

                # Realizar movimiento (alternar entre jugada óptima y jugada aleatoria para variedad)
                if random.random() < 0.40:
                    sim_move = random.choice(sim_model.get_available_moves())
                else:
                    sim_move = best_move

                sim_model.make_move(sim_move)

            if g % step == 0 or g == num_games:
                print(f"  -> Progreso: Partida {g:,}/{num_games:,} ({(g / num_games) * 100:.0f}%) | Estados únicos descubiertos: {len(dataset_records):,}")

        # Guardar en archivo CSV
        headers = [f"cell_{i}" for i in range(9)] + ["best_move"]
        with open(self.dataset_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for board_features, target in dataset_records:
                writer.writerow(board_features + [target])

        elapsed_gen = time.perf_counter() - start_gen_time
        print(f"[ML Agent] Dataset guardado exitosamente en '{self.dataset_path}' ({len(dataset_records):,} ejemplos generados en {elapsed_gen:.2f}s).\n")
        return self.dataset_path

    def train(self) -> None:
        """Carga el dataset (o lo genera si no existe) y entrena el Árbol Binario de Decisión."""
        if not os.path.exists(self.dataset_path):
            self.generate_dataset(num_games=1000)

        print(f"[ML Agent] Entrenando Árbol Binario de Decisión (Criterio: Impureza de Gini)...")
        start_train_time = time.perf_counter()

        X: List[List[float]] = []
        y: List[int] = []

        with open(self.dataset_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            for row in reader:
                if not row:
                    continue
                features = [float(val) for val in row[:9]]
                target = int(row[9])
                X.append(features)
                y.append(target)

        self.tree.fit(X, y)
        self.is_trained = True

        elapsed_train = (time.perf_counter() - start_train_time) * 1000
        print(f"[ML Agent] Entrenamiento completado con éxito:")
        print(f"  -> Muestras de entrenamiento: {len(X):,}")
        print(f"  -> Profundidad máxima del árbol: {self.tree.max_depth}")
        print(f"  -> Precisión alcanzada en dataset: {self.tree.accuracy:.2f}%")
        print(f"  -> Tiempo de entrenamiento: {elapsed_train:.2f} ms")

        # Exportar gráfico inicial
        plot_path = "data/decision_tree.png"
        self.tree.export_plot(plot_path)
        print(f"[ML Agent] Diagrama visual exportado a '{plot_path}'.\n")

    def get_best_move(self, model: GameModel) -> Optional[int]:
        """Calcula la mejor jugada usando inferencia instantánea del Árbol Binario."""
        available_moves = model.get_available_moves()
        if not available_moves:
            return None

        if not self.is_trained:
            self.train()

        start_time = time.perf_counter()
        encoded = self.encode_board(model.get_board())
        predicted_move = self.tree.predict_row(encoded)
        elapsed = (time.perf_counter() - start_time) * 1000
        self.last_response_time_ms = round(elapsed, 4)

        # Si la casilla predicha es válida y está libre, usarla
        if predicted_move in available_moves:
            return predicted_move

        # Si el árbol recomienda una casilla ya ocupada, fallback a la primera libre
        return random.choice(available_moves)
