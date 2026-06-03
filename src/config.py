"""Constantes del contrato de evaluación y configuración de mapas (Fase 2).

Centraliza los valores que NO deben divergir entre entrenamiento y evaluación:
forma de observación, número de frames apilados, mapas permitidos y el mapa de
evaluación oculto (que jamás debe usarse para entrenar).
"""

from __future__ import annotations

# --- Contrato de observación -------------------------------------------------
# El modelo DEBE esperar (1, 64, 64) por frame y (4, 64, 64) tras FrameStack.
OBS_SHAPE: tuple[int, int, int] = (1, 64, 64)
N_STACK: int = 4
STACKED_SHAPE: tuple[int, int, int] = (N_STACK, OBS_SHAPE[1], OBS_SHAPE[2])  # (4, 64, 64)

# --- Mapas -------------------------------------------------------------------
# Mapas permitidos para ENTRENAR (multi-mapa para forzar generalización).
TRAIN_MAPS: list[str] = [
    "Duckietown-loop_empty-v0",
    "Duckietown-udem1-v0",
    "Duckietown-zigzag_dists-v0",
    "Duckietown-small_loop-v0",
    "Duckietown-straight_road-v0",
]

# Mapa OCULTO de evaluación. PROHIBIDO entrenar aquí (descalificación).
# Aislado a propósito: nunca se incluye en TRAIN_MAPS y make_env lo bloquea.
EVAL_MAP: str = "Duckietown-loop_obstacles-v0"

# --- Acciones discretas (solo DQN) -------------------------------------------
# Cada entrada es un comando continuo [velocidad, giro] en el rango [-1, 1].
# DQN no maneja acciones continuas: este mapeo discreto -> continuo lo resuelve
# DiscreteActionWrapper. PPO/SAC/TD3 usan el Box continuo directamente.
DISCRETE_ACTIONS: list[list[float]] = [
    [0.6, 0.0],   # 0: avanzar recto
    [0.4, -0.5],  # 1: girar izquierda suave
    [0.4, 0.5],   # 2: girar derecha suave
    [0.3, -1.0],  # 3: girar izquierda fuerte
    [0.3, 1.0],   # 4: girar derecha fuerte
]

# --- CNN ---------------------------------------------------------------------
FEATURES_DIM: int = 256
