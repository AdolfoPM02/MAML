"""Diagnóstico de runtime de Duckietown (aislar segfaults nativos).

El entrenamiento PPO real en Colab crashea con `Segmentation fault (core dumped)`
incluso forzando CPU. Eso apunta a un fallo NATIVO (Duckietown/OpenGL/pyglet/xvfb)
y no a CUDA. Este script ejecuta niveles crecientes de aislamiento para localizar
EN QUÉ FASE ocurre el crash:

  A) base            : entorno base (make_base_env) + reset + N steps aleatorios.
  B) --vec           : build_vec_env + VecFrameStack(4) + reset + N steps.
  C) --sb3-init      : construir PPO con CustomCNN (sin learn).
  D) --sb3-learn     : construir PPO + learn(total_timesteps) muy corto.

Cada modo imprime claramente la fase (reset / step loop / vec reset / sb3 init /
sb3 learn), de modo que si hay un segfault se ve cuál fue la última fase alcanzada.

Ejemplos (en Colab, bajo xvfb):
    xvfb-run -a PY scripts/debug_duckie_runtime.py --steps 20 --device cpu
    xvfb-run -a PY scripts/debug_duckie_runtime.py --vec --steps 20 --device cpu
    xvfb-run -a PY scripts/debug_duckie_runtime.py --sb3-init --device cpu
    xvfb-run -a PY scripts/debug_duckie_runtime.py --sb3-learn --timesteps 64 --device cpu

En local (sin Duckietown) para comprobar que el script es correcto: --use-mock.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import config
from src.duckie_factory import gym_duckietown_available, make_base_env

FLUSH = dict(flush=True)


def log(phase: str, msg: str = "") -> None:
    print(f"[fase: {phase}] {msg}", **FLUSH)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnóstico de runtime de Duckietown.")
    p.add_argument("--use-mock", action="store_true",
                   help="Forzar entorno mock (lint local sin Duckietown).")
    p.add_argument("--map", default=config.TRAIN_MAPS[0],
                   help="Mapa de entrenamiento (nombre exacto de TRAIN_MAPS).")
    p.add_argument("--steps", type=int, default=50,
                   help="Número de steps aleatorios en los modos A y B.")
    p.add_argument("--vec", action="store_true",
                   help="Modo B: build_vec_env + VecFrameStack.")
    p.add_argument("--sb3-init", action="store_true",
                   help="Modo C: construir PPO con CustomCNN (sin learn).")
    p.add_argument("--sb3-learn", action="store_true",
                   help="Modo D: PPO.learn con timesteps muy bajos.")
    p.add_argument("--timesteps", type=int, default=64,
                   help="total_timesteps para --sb3-learn (default 64).")
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    return p.parse_args(argv)


def _reject_eval_map(map_name: str) -> None:
    if map_name == config.EVAL_MAP:
        raise ValueError(
            f"'{map_name}' es el mapa de EVALUACIÓN oculto: prohibido usarlo aquí."
        )


def _sample_base_action(env) -> np.ndarray:
    """Acción aleatoria para el entorno base (Box [v, steering] si existe)."""
    space = getattr(env, "action_space", None)
    if space is not None:
        return space.sample()
    return np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)


def mode_base(args) -> None:
    use_mock = True if args.use_mock else None
    log("init", f"modo A (base) | map={args.map} | mock={args.use_mock}")
    env = make_base_env(args.map, use_mock=use_mock)
    try:
        log("reset", "reset() del entorno base...")
        obs = np.asarray(env.reset())
        log("reset", f"obs cruda {obs.shape}")
        log("step loop", f"ejecutando {args.steps} steps aleatorios...")
        for i in range(args.steps):
            action = _sample_base_action(env)
            obs, reward, done, info = env.step(action)
            if i < 3 or i == args.steps - 1:
                log("step loop", f"step={i} reward={float(reward):.3f} done={bool(done)}")
            if done:
                log("step loop", f"done en step {i}, reset()")
                env.reset()
        log("done", "modo A completado sin crash")
    finally:
        if hasattr(env, "close"):
            env.close()


def mode_vec(args) -> None:
    from src.envs import build_vec_env
    use_mock = True if args.use_mock else None
    log("init", f"modo B (vec) | map={args.map} | mock={args.use_mock}")
    venv = build_vec_env([args.map], discrete=False, use_mock=use_mock,
                         n_stack=config.N_STACK, allow_eval=False)
    try:
        log("vec reset", f"obs_space {venv.observation_space.shape}")
        obs = venv.reset()
        log("vec reset", f"obs {obs.shape}")
        assert obs.shape == (1, *config.STACKED_SHAPE), f"obs inesperada {obs.shape}"
        log("step loop", f"ejecutando {args.steps} steps aleatorios...")
        for i in range(args.steps):
            action = np.array([venv.action_space.sample()])
            obs, rewards, dones, infos = venv.step(action)
            if i < 3 or i == args.steps - 1:
                log("step loop", f"step={i} obs={obs.shape} reward={float(rewards[0]):.3f} done={bool(dones[0])}")
        log("done", "modo B completado sin crash")
    finally:
        venv.close()


def _make_ppo(args):
    from stable_baselines3 import PPO
    from src.cnn import CustomCNN
    from src.envs import build_vec_env
    use_mock = True if args.use_mock else None
    log("vec reset", "construyendo build_vec_env real...")
    venv = build_vec_env([args.map], discrete=False, use_mock=use_mock,
                         n_stack=config.N_STACK, allow_eval=False)
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=config.FEATURES_DIM),
    )
    log("sb3 init", f"construyendo PPO (device={args.device})...")
    model = PPO("CnnPolicy", venv, policy_kwargs=policy_kwargs,
                n_steps=64, batch_size=64, n_epochs=1,
                device=args.device, verbose=1)
    log("sb3 init", "PPO construido OK")
    return model, venv


def mode_sb3_init(args) -> None:
    log("init", f"modo C (sb3-init) | map={args.map} | mock={args.use_mock}")
    model, venv = _make_ppo(args)
    try:
        log("done", "modo C completado: PPO inicializado sin entrenar")
    finally:
        venv.close()


def mode_sb3_learn(args) -> None:
    log("init", f"modo D (sb3-learn) | map={args.map} | timesteps={args.timesteps}")
    model, venv = _make_ppo(args)
    tmp = os.path.join("models", "_debug_ppo")
    tmp_zip = tmp + ".zip"
    try:
        log("sb3 learn", f"PPO.learn(total_timesteps={args.timesteps})...")
        model.learn(total_timesteps=args.timesteps)
        log("sb3 learn", "learn() terminó sin crash")
        os.makedirs("models", exist_ok=True)
        model.save(tmp)
        log("done", f"modo D completado; modelo temporal guardado y se borrará")
    finally:
        venv.close()
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)


def main(argv=None) -> None:
    args = parse_args(argv)
    _reject_eval_map(args.map)
    print("=" * 64, **FLUSH)
    print(f"DEBUG DUCKIETOWN RUNTIME | gym_duckietown disponible: "
          f"{gym_duckietown_available()}", **FLUSH)
    print("=" * 64, **FLUSH)

    if args.sb3_learn:
        mode_sb3_learn(args)
    elif args.sb3_init:
        mode_sb3_init(args)
    elif args.vec:
        mode_vec(args)
    else:
        mode_base(args)

    print("=" * 64, **FLUSH)
    print("DEBUG DUCKIETOWN: FIN OK", **FLUSH)
    print("=" * 64, **FLUSH)


if __name__ == "__main__":
    main()
