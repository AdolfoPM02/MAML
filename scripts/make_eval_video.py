"""Genera un VÍDEO MP4 demostrativo del modelo final conduciendo en Duckietown.

Es una utilidad CUALITATIVA y OPCIONAL: la evaluación cuantitativa oficial sigue siendo
`eval.py`. Este script carga un modelo (.zip) entrenado con `train.py`, reconstruye el
entorno IDÉNTICO al de entrenamiento (`build_vec_env` + `VecFrameStack(4)` → obs
(4,64,64)) y graba el agente conduciendo, guardando un .mp4.

Por defecto graba UN ÚNICO rollout reproducible (con `--seed`, default 42). De forma
opcional, con `--rollouts > 1` corre varios rollouts y guarda el de MAYOR recompensa
acumulada (útil para descartar un rollout degenerado, p. ej. el coche parado).

No entrena nada. Reutiliza el patrón `model-first` de `eval.py` (cargar el modelo sobre
un entorno sintético y luego `set_env(real)`) para evitar el segfault de SB3 + Duckietown.

Ejemplos:
    # Local sin Duckietown (mock): valida el pipeline y produce un mp4 de ruido.
    python scripts/make_eval_video.py --use-mock --model models/best_agent \
        --out outputs/best_agent_loop_empty.mp4 --rollouts 2 --max-steps 100

    # Colab con Duckietown real (headless):
    xvfb-run -a python scripts/make_eval_video.py --model models/best_agent \
        --map Duckietown-loop_empty-v0 --out outputs/best_agent_loop_empty.mp4 --rollouts 1 --seed 42
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from stable_baselines3 import DQN, PPO, SAC
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from src import config
from src.envs import build_vec_env

# ppo_adv / ppo_adv_v2 (Fase 3) cargan con la clase PPO (PPO con hiperparámetros avanzados).
ALGO_CLASSES = {"dqn": DQN, "ppo": PPO, "ppo_adv": PPO, "ppo_adv_v2": PPO, "sac": SAC}


def set_global_seeds(seed: int) -> None:
    """Fija la semilla en random, numpy y torch (reproducibilidad razonable; NO promete
    resultados bit a bit, sobre todo en GPU o con el simulador Duckietown)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _PlaceholderEnv(gym.Env):
    """Entorno sintético (sin Duckietown) con los MISMOS espacios que el real tras
    FrameStack, para el flujo `model-first`: el modelo se carga sobre este placeholder
    ANTES de crear Duckietown, y luego se hace `set_env(real)`."""

    metadata = {"render_modes": []}

    def __init__(self, discrete: bool, n_stack: int, max_steps: int = 100):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(n_stack, 64, 64), dtype=np.uint8)
        if discrete:
            self.action_space = spaces.Discrete(len(config.DISCRETE_ACTIONS))
        else:
            # Debe coincidir con DuckieWrapper: velocidad [0,1], giro [-1,1].
            self.action_space = spaces.Box(
                low=np.array([0.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32)
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
    p = argparse.ArgumentParser(
        description="Genera un vídeo MP4 del modelo final conduciendo en Duckietown.")
    p.add_argument("--algo", default="ppo",
                   choices=["dqn", "ppo", "ppo_adv", "ppo_adv_v2", "sac"],
                   help="Algoritmo del modelo (elige la clase para .load). Default: ppo.")
    p.add_argument("--model", default="models/best_agent",
                   help="Ruta al .zip del modelo (con o sin extensión).")
    p.add_argument("--map", default="Duckietown-loop_empty-v0",
                   help="Mapa de TRAIN_MAPS, o el de evaluación oculto (solo --allow-eval).")
    p.add_argument("--out", default="outputs/best_agent_loop_empty.mp4",
                   help="Ruta de salida del .mp4 (se crea el directorio si falta).")
    p.add_argument("--rollouts", type=int, default=1,
                   help="Nº de rollouts. Default 1 = un único rollout reproducible (con "
                        "--seed). Solo si --rollouts > 1 se prueban varios y se guarda el "
                        "de MAYOR recompensa.")
    p.add_argument("--max-steps", type=int, default=1500,
                   help="Límite de pasos por rollout (corta rollouts que no terminan).")
    p.add_argument("--fps", type=int, default=30, help="FPS del vídeo de salida.")
    p.add_argument("--use-mock", action="store_true",
                   help="Forzar entorno mock (sin Duckietown; produce ruido, solo prueba).")
    p.add_argument("--allow-eval", action="store_true",
                   help="Habilita Duckietown-loop_obstacles-v0 SOLO PARA EVALUACIÓN.")
    p.add_argument("--disable-movement-shaping", action="store_true",
                   help="Graba con la recompensa LIMPIA del simulador (sin el reward "
                        "shaping de movimiento del DuckieWrapper).")
    p.add_argument("--min-forward-speed", type=float, default=0.0,
                   help="Velocidad mínima de avance forzada (0 = sin forzar). >0 (p. ej. "
                        "0.1) permite grabar conducción real con la misma restricción "
                        "usada en entrenamiento.")
    p.add_argument("--seed", type=int, default=42,
                   help="Semilla para random/numpy/torch y el entorno.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--n-stack", type=int, default=config.N_STACK)
    p.add_argument("--deterministic", dest="deterministic", action="store_true",
                   default=True, help="Política determinista (default).")
    p.add_argument("--stochastic", dest="deterministic", action="store_false",
                   help="Política estocástica.")
    return p.parse_args(argv)


def _base_env(vec_env):
    """Devuelve el DuckieWrapper subyacente (para render RGB): VecFrameStack -> venv -> envs[0]."""
    return vec_env.venv.envs[0]


def _run_rollout(model, vec_env, max_steps: int, deterministic: bool):
    """Corre un rollout y devuelve (frames RGB, recompensa acumulada)."""
    frames = []
    obs = vec_env.reset()
    base = _base_env(vec_env)
    frames.append(np.asarray(base.render(), dtype=np.uint8))
    total = 0.0
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, _ = vec_env.step(action)
        total += float(rewards[0])
        frames.append(np.asarray(base.render(), dtype=np.uint8))
        if bool(dones[0]):
            break
    return frames, total


def _write_video(path: str, frames, fps: int) -> None:
    """Escribe los frames a un .mp4 con imageio (macro_block_size=1 admite dimensiones impares)."""
    import imageio
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, macro_block_size=1)


def make_video(args: argparse.Namespace) -> dict:
    discrete = args.algo == "dqn"
    cls = ALGO_CLASSES[args.algo]

    # model-first: cargar el modelo sobre un env sintético y luego set_env(real).
    placeholder = DummyVecEnv([lambda: _PlaceholderEnv(discrete, args.n_stack)])
    model = cls.load(args.model, env=placeholder, device=args.device)
    vec_env = build_vec_env([args.map], discrete=discrete,
                            use_mock=(args.use_mock or None), seed=args.seed,
                            n_stack=args.n_stack, allow_eval=args.allow_eval,
                            enable_movement_shaping=not args.disable_movement_shaping,
                            min_forward_speed=args.min_forward_speed)
    model.set_env(vec_env)
    placeholder.close()

    best_frames, best_reward, best_idx = None, -np.inf, -1
    for i in range(args.rollouts):
        frames, reward = _run_rollout(model, vec_env, args.max_steps, args.deterministic)
        print(f"  rollout {i + 1}/{args.rollouts}: reward={reward:.3f} | frames={len(frames)}")
        if reward > best_reward:
            best_frames, best_reward, best_idx = frames, reward, i
    vec_env.close()

    _write_video(args.out, best_frames, args.fps)
    return {
        "out": args.out,
        "best_reward": float(best_reward),
        "best_rollout": int(best_idx) + 1,
        "frames": len(best_frames),
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    set_global_seeds(args.seed)
    print("=" * 64)
    print(f"VIDEO | algo={args.algo} | model={args.model} | map={args.map}")
    print(f"      | rollouts={args.rollouts} | seed={args.seed} | mock={args.use_mock} | "
          f"allow_eval={args.allow_eval} | out={args.out}")
    print("=" * 64)

    m = make_video(args)
    print("=" * 64)
    print(f"Mejor rollout      : #{m['best_rollout']} (recompensa {m['best_reward']:.3f})")
    print(f"Frames guardados   : {m['frames']}")
    print(f"Vídeo escrito en   : {m['out']}")
    print("=" * 64)


if __name__ == "__main__":
    main()
