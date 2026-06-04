"""Evaluación de un modelo Duckietown guardado (Fase de evaluación).

Carga un modelo (.zip) entrenado con `train.py`, reconstruye el entorno IDÉNTICO al de
entrenamiento (`build_vec_env` + `VecFrameStack(4)` → observación (4,64,64)) y reporta
métricas simples sobre N episodios.

Local (sin Duckietown): `--use-mock`. En Colab (con gym_duckietown) el MISMO comando,
sin `--use-mock`, evalúa en Duckietown real (autodetección en `make_base_env`).

El mapa oculto `Duckietown-loop_obstacles-v0` está bloqueado salvo que se pase
`--allow-eval` (que SOLO habilita evaluación, nunca entrenamiento).

Ejemplos:
    python eval.py --algo ppo --use-mock --model models/_evtest_ppo --map Duckietown-loop_empty-v0 --episodes 3
    python eval.py --algo sac --model models/best_duckie_agent --map Duckietown-loop_obstacles-v0 --allow-eval
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import DQN, PPO, SAC
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from src import config
from src.envs import build_vec_env

ALGO_CLASSES = {"dqn": DQN, "ppo": PPO, "sac": SAC}


class _PlaceholderEnv(gym.Env):
    """Entorno sintético (sin Duckietown) con los MISMOS espacios que el entorno real
    tras FrameStack, usado por el flujo `model-first`: el modelo se carga sobre este
    placeholder ANTES de crear Duckietown, y luego se hace `set_env(real)`.

    obs    = Box(0, 255, (n_stack, 64, 64), uint8)
    action = Discrete(len(DISCRETE_ACTIONS))    si discrete (DQN)
             Box(-1, 1, (2,), float32)          si continuo (PPO/SAC)
    """

    metadata = {"render_modes": []}

    def __init__(self, discrete: bool, n_stack: int, max_steps: int = 100):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(n_stack, 64, 64), dtype=np.uint8)
        if discrete:
            self.action_space = spaces.Discrete(len(config.DISCRETE_ACTIONS))
        else:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.max_steps = max_steps
        self._n = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._n = 0
        return self.observation_space.sample(), {}

    def step(self, action):
        self._n += 1
        return self.observation_space.sample(), 0.0, False, self._n >= self.max_steps, {}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluar un modelo Duckietown guardado.")
    p.add_argument("--algo", required=True, choices=["dqn", "ppo", "sac"],
                   help="Algoritmo del modelo (elige la clase para .load).")
    p.add_argument("--model", required=True,
                   help="Ruta al .zip del modelo (con o sin extensión).")
    p.add_argument("--map", default="Duckietown-loop_empty-v0",
                   help="Nombre exacto de TRAIN_MAPS, o el mapa de evaluación oculto "
                        "(solo con --allow-eval).")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--use-mock", action="store_true",
                   help="Forzar entorno mock (evaluación local sin Duckietown).")
    p.add_argument("--allow-eval", action="store_true",
                   help="Habilita Duckietown-loop_obstacles-v0 SOLO PARA EVALUACIÓN, "
                        "NUNCA para entrenamiento. Sin este flag, ese mapa está "
                        "bloqueado por el guard de make_env.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--init-order", default="env-first",
                   choices=["env-first", "model-first"],
                   help="env-first (default): crea el env real y luego carga el modelo. "
                        "model-first: carga el modelo sobre un env sintético y luego "
                        "set_env(real) — evita el segfault de SB3 + Duckietown real.")
    p.add_argument("--n-stack", type=int, default=config.N_STACK)
    p.add_argument("--deterministic", dest="deterministic", action="store_true",
                   default=True, help="Política determinista (default).")
    p.add_argument("--stochastic", dest="deterministic", action="store_false",
                   help="Política estocástica.")
    p.add_argument("--success-min-len", type=int, default=None,
                   help="Umbral de longitud para contar un episodio como 'éxito' "
                        "(proxy de supervivencia). Si se omite, éxito = N/A.")
    return p.parse_args(argv)


def evaluate(args: argparse.Namespace) -> dict:
    discrete = args.algo == "dqn"
    cls = ALGO_CLASSES[args.algo]

    if args.init_order == "model-first":
        # Cargar el modelo sobre un env SINTÉTICO (sin Duckietown) y luego set_env(real).
        # Evita el segfault de cargar/usar SB3 con Duckietown real directamente.
        placeholder = DummyVecEnv(
            [lambda: _PlaceholderEnv(discrete, args.n_stack)])
        model = cls.load(args.model, env=placeholder, device=args.device)
        env = build_vec_env([args.map], discrete=discrete,
                            use_mock=(args.use_mock or None), seed=args.seed,
                            n_stack=args.n_stack, allow_eval=args.allow_eval)
        model.set_env(env)
        placeholder.close()
    else:
        # env-first (default): crear el entorno real y cargar el modelo con él.
        env = build_vec_env([args.map], discrete=discrete,
                            use_mock=(args.use_mock or None), seed=args.seed,
                            n_stack=args.n_stack,
                            allow_eval=args.allow_eval)  # GUARD: bloquea EVAL_MAP sin allow_eval
        model = cls.load(args.model, env=env, device=args.device)

    rewards, lengths = evaluate_policy(
        model, env, n_eval_episodes=args.episodes,
        deterministic=args.deterministic, return_episode_rewards=True,
    )
    env.close()

    rewards = np.asarray(rewards, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    metrics = {
        "episodes": int(args.episodes),
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "mean_length": float(lengths.mean()),
    }
    if args.success_min_len is not None:
        metrics["success_rate"] = float((lengths >= args.success_min_len).mean())
    else:
        metrics["success_rate"] = None
    return metrics


def main(argv=None) -> None:
    args = parse_args(argv)
    print("=" * 64)
    print(f"EVAL | algo={args.algo} | model={args.model} | map={args.map}")
    print(f"     | episodes={args.episodes} | mock={args.use_mock} | "
          f"allow_eval={args.allow_eval} | deterministic={args.deterministic} | "
          f"init-order={args.init_order}")
    print("=" * 64)

    m = evaluate(args)
    print(f"Recompensa acumulada media : {m['mean_reward']:.3f} ± {m['std_reward']:.3f}")
    print(f"Longitud media de episodio : {m['mean_length']:.1f}")
    if m["success_rate"] is None:
        print(f"Tasa de éxito              : N/A (usa --success-min-len para calcularla)")
    else:
        print(f"Tasa de éxito (len>={args.success_min_len}) : {m['success_rate']:.3f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
