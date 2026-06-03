"""Fase 1 — Q-Learning tabular "desde cero" sobre FrozenLake-v1 (8x8, resbaladiza).

Proyecto final de Aprendizaje por Refuerzo (Duckietown). Esta fase demuestra la
comprensión de la Ecuación de Bellman implementando Q-Learning tabular sin ninguna
librería de Deep RL: solo `gymnasium` como entorno, `numpy` para la Q-table y
`matplotlib` para la gráfica de progreso.

Ejecución (entorno del proyecto, Python 3.11):
    .venv-maml/bin/python q_learning_sandbox.py

Salida:
    - Por consola: hiperparámetros, recompensa media móvil final y tasa de éxito
      de la política greedy aprendida.
    - Fichero: outputs/q_learning_frozenlake.png (curva de media móvil de recompensa).
"""

from __future__ import annotations

import os

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")  # backend sin display (WSL/servidor): guardamos a fichero.
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# Hiperparámetros (configurables)
# --------------------------------------------------------------------------- #
EPISODES = 50_000          # 8x8 resbaladiza es difícil; Q-learning vanilla necesita
                           # bastantes episodios para converger de forma estable.
ALPHA = 0.1                # tasa de aprendizaje (step-size).
GAMMA = 0.99               # factor de descuento.
EPS_START = 1.0            # exploración inicial (100% aleatorio).
EPS_MIN = 0.05             # exploración mínima residual.
EPS_DECAY_FRAC = 0.6       # epsilon decae linealmente hasta EPS_MIN cubriendo
                           # esta fracción de los episodios.
MOVING_AVG_WINDOW = 100    # ventana de la media móvil (métrica que pide el reto).
SEED = 42                  # reproducibilidad.

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
PLOT_PATH = os.path.join(OUT_DIR, "q_learning_frozenlake.png")


def make_env() -> gym.Env:
    """Crea el entorno FrozenLake-v1 8x8 resbaladizo."""
    return gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)


def epsilon_greedy(Q: np.ndarray, state: int, epsilon: float,
                   n_actions: int, rng: np.random.Generator) -> int:
    """Política epsilon-greedy: explora con prob. epsilon, explota el resto."""
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))
    # Desempate aleatorio entre acciones de máximo valor (evita sesgo hacia a=0).
    q_s = Q[state]
    best = np.flatnonzero(q_s == q_s.max())
    return int(rng.choice(best))


def train_q_learning(episodes: int = EPISODES,
                     alpha: float = ALPHA,
                     gamma: float = GAMMA,
                     eps_start: float = EPS_START,
                     eps_min: float = EPS_MIN,
                     eps_decay_frac: float = EPS_DECAY_FRAC,
                     seed: int = SEED) -> tuple[np.ndarray, list[float]]:
    """Entrena Q-Learning tabular con actualización de Bellman.

    Q[s,a] <- Q[s,a] + alpha * (r + gamma * max_a' Q[s',a'] - Q[s,a])

    Devuelve la Q-table aprendida y la lista de recompensas por episodio
    (1.0 si el agente alcanza la meta, 0.0 en caso contrario).
    """
    env = make_env()
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    rng = np.random.default_rng(seed)
    Q = np.zeros((n_states, n_actions), dtype=np.float64)

    rewards: list[float] = []
    decay_episodes = max(1, int(episodes * eps_decay_frac))

    for ep in range(episodes):
        # Decaimiento lineal de epsilon desde eps_start hasta eps_min.
        epsilon = max(eps_min,
                      eps_start - (eps_start - eps_min) * (ep / decay_episodes))

        state, _ = env.reset(seed=seed + ep)
        terminated = truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            action = epsilon_greedy(Q, state, epsilon, n_actions, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Ecuación de Bellman (TD target con bootstrapping).
            best_next = np.max(Q[next_state])
            td_target = reward + gamma * best_next * (not terminated)
            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

    env.close()
    return Q, rewards


def moving_average(data: list[float] | np.ndarray, window: int) -> np.ndarray:
    """Media móvil simple (modo 'valid')."""
    data = np.asarray(data, dtype=np.float64)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def evaluate_policy(Q: np.ndarray, episodes: int = 1000,
                    seed: int = SEED) -> float:
    """Tasa de éxito de la política greedy (sin exploración) aprendida."""
    env = make_env()
    rng = np.random.default_rng(seed + 1)
    successes = 0

    for ep in range(episodes):
        state, _ = env.reset(seed=seed + 100_000 + ep)
        terminated = truncated = False
        while not (terminated or truncated):
            q_s = Q[state]
            best = np.flatnonzero(q_s == q_s.max())
            action = int(rng.choice(best))  # greedy con desempate aleatorio
            state, reward, terminated, truncated, _ = env.step(action)
        successes += int(reward > 0)

    env.close()
    return successes / episodes


def plot_results(rewards: list[float], window: int, out_path: str) -> None:
    """Guarda la curva de media móvil de recompensa por episodio."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ma = moving_average(rewards, window)

    plt.figure(figsize=(10, 5))
    plt.plot(ma)
    plt.title(f"Progreso Q-Learning — FrozenLake-v1 8x8 (media móvil, ventana={window})")
    plt.xlabel("Episodio")
    plt.ylabel(f"Tasa de éxito (media de {window} episodios)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main() -> None:
    print("=" * 60)
    print("Fase 1 — Q-Learning tabular | FrozenLake-v1 (8x8, slippery)")
    print("=" * 60)
    print(f"Episodios={EPISODES}  alpha={ALPHA}  gamma={GAMMA}")
    print(f"epsilon: {EPS_START} -> {EPS_MIN} (decay lineal en {EPS_DECAY_FRAC:.0%} de episodios)")
    print(f"Semilla={SEED}")
    print("-" * 60)

    Q, rewards = train_q_learning()

    ma = moving_average(rewards, MOVING_AVG_WINDOW)
    final_ma = float(ma[-1]) if len(ma) else float("nan")
    success_rate = evaluate_policy(Q, episodes=1000)

    print(f"Recompensa media móvil final (ventana {MOVING_AVG_WINDOW}): {final_ma:.3f}")
    print(f"Tasa de éxito política greedy (1000 episodios):       {success_rate:.3f}")

    plot_results(rewards, MOVING_AVG_WINDOW, PLOT_PATH)
    print(f"Gráfica guardada en: {PLOT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
