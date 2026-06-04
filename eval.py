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
from stable_baselines3.common.evaluation import evaluate_policy

from src import config
from src.envs import build_vec_env

ALGO_CLASSES = {"dqn": DQN, "ppo": PPO, "sac": SAC}


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
    env = build_vec_env([args.map], discrete=discrete, use_mock=(args.use_mock or None),
                        seed=args.seed, n_stack=args.n_stack,
                        allow_eval=args.allow_eval)  # GUARD: bloquea EVAL_MAP sin allow_eval

    model = ALGO_CLASSES[args.algo].load(args.model, env=env, device=args.device)

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
          f"allow_eval={args.allow_eval} | deterministic={args.deterministic}")
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
