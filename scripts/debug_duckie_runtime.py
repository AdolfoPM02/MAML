"""Diagnóstico de runtime de Duckietown (aislar segfaults nativos).

El entrenamiento PPO real en Colab crashea con `Segmentation fault (core dumped)`.
Diagnósticos previos acotaron la causa: el crash ocurre al **construir/inicializar PPO
con el entorno Duckietown real** bajo xvfb/pyglet/OpenGL. NO es CUDA, NO es CustomCNN,
NO son los spaces, NO es VecFrameStack, NO es learn (un entorno sintético con espacios
idénticos inicializa PPO sin problema).

Niveles de aislamiento (de menor a mayor):
  A) base            : entorno base (make_base_env) + reset + N steps aleatorios.
  B) --vec           : build_vec_env + VecFrameStack(4) + reset + N steps.
  --check-spaces      : build_vec_env real -> imprime spaces/dtype/low/high/reset (sin PPO).
  C) --sb3-init      : construir PPO (sin learn).
  D) --sb3-learn     : construir PPO + learn(total_timesteps) corto.

Opciones:
  --synthetic-env    : entorno Gymnasium SINTÉTICO (NO Duckietown) con
                       obs Box(0,255,(4,64,64),uint8) y acción Box([0,-1],[1,1]).
  --no-custom-cnn    : PPO con "CnnPolicy" por defecto (NatureCNN), sin CustomCNN.
  --mlp-policy       : PPO con "MlpPolicy" (solo con --synthetic-env o --use-mock).
  --init-order       : env-first (default) construye env real y luego PPO;
                       model-first construye PPO sobre un env sintético compatible y
                       luego hace set_env() con el entorno real (prueba si cambiar el
                       orden torch/SB3 <-> Duckietown evita el segfault).

Backends OpenGL: NO se fijan en Python; se prueban como variables de entorno desde el
notebook (PYOPENGL_PLATFORM=egl | osmesa | sin definir). Ver sección 8C de COLAB_SETUP.md.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src import config
from src.duckie_factory import gym_duckietown_available, make_base_env

FLUSH = dict(flush=True)

# Bounds del action_space del DuckieWrapper real (src/wrappers.py): Box([-1,-1],[1,1]).
# El placeholder de model-first debe coincidir EXACTAMENTE para que set_env() valide.
DUCKIE_ACTION_LOW = np.array([-1.0, -1.0], dtype=np.float32)
DUCKIE_ACTION_HIGH = np.array([1.0, 1.0], dtype=np.float32)


def log(phase: str, msg: str = "") -> None:
    print(f"[fase: {phase}] {msg}", **FLUSH)


class SyntheticEnv(gym.Env):
    """Entorno Gymnasium sintético con los MISMOS espacios que Duckietown tras
    FrameStack, pero SIN importar ni usar Duckietown/OpenGL.

    obs = Box(0, 255, (4, 64, 64), uint8). Acción configurable:
    - por defecto Box([0,-1],[1,1]) (spec de --synthetic-env);
    - duckie_action=True -> Box([-1,-1],[1,1]) para casar con el DuckieWrapper real
      (necesario en model-first para que set_env() valide los espacios).
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 100, duckie_action: bool = False):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0, high=255, shape=config.STACKED_SHAPE, dtype=np.uint8)
        if duckie_action:
            low, high = DUCKIE_ACTION_LOW, DUCKIE_ACTION_HIGH
        else:
            low = np.array([0.0, -1.0], dtype=np.float32)
            high = np.array([1.0, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.max_steps = max_steps
        self._n = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._n = 0
        return self.observation_space.sample(), {}

    def step(self, action):
        self._n += 1
        obs = self.observation_space.sample()
        truncated = self._n >= self.max_steps
        return obs, 0.0, False, truncated, {"synthetic": True}


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
    p.add_argument("--check-spaces", action="store_true",
                   help="Imprime spaces/dtype/low/high/reset del entorno (sin PPO).")
    p.add_argument("--sb3-init", action="store_true",
                   help="Construir PPO (sin learn).")
    p.add_argument("--sb3-learn", action="store_true",
                   help="PPO.learn con timesteps muy bajos.")
    p.add_argument("--timesteps", type=int, default=64,
                   help="total_timesteps para --sb3-learn (default 64).")
    p.add_argument("--synthetic-env", action="store_true",
                   help="Usar entorno sintético (NO Duckietown) con espacios idénticos.")
    p.add_argument("--no-custom-cnn", action="store_true",
                   help="PPO con CnnPolicy por defecto, sin CustomCNN.")
    p.add_argument("--mlp-policy", action="store_true",
                   help="PPO con MlpPolicy (solo con --synthetic-env o --use-mock).")
    p.add_argument("--init-order", default="env-first",
                   choices=["env-first", "model-first"],
                   help="env-first: env real -> PPO. model-first: PPO sobre env "
                        "sintético -> set_env(real).")
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    return p.parse_args(argv)


def _reject_eval_map(map_name: str) -> None:
    if map_name == config.EVAL_MAP:
        raise ValueError(
            f"'{map_name}' es el mapa de EVALUACIÓN oculto: prohibido usarlo aquí."
        )


def _sample_base_action(env) -> np.ndarray:
    space = getattr(env, "action_space", None)
    if space is not None:
        return space.sample()
    return np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)


def _build_real_env(args):
    """build_vec_env real/mock (Duckietown). NO sintético."""
    from src.envs import build_vec_env
    use_mock = True if args.use_mock else None
    return build_vec_env([args.map], discrete=False, use_mock=use_mock,
                         n_stack=config.N_STACK, allow_eval=False)


def _build_sb3_env(args):
    """VecEnv para los modos sb3: sintético (sin Duckietown) o real/mock."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    if args.synthetic_env:
        log("vec reset", "construyendo entorno SINTÉTICO (sin Duckietown)...")
        return DummyVecEnv([lambda: SyntheticEnv()])
    log("vec reset", "construyendo build_vec_env (Duckietown real/mock)...")
    return _build_real_env(args)


def _describe_policy(args) -> str:
    if args.mlp_policy:
        return "MlpPolicy (sin CustomCNN)"
    if args.no_custom_cnn:
        return "CnnPolicy por defecto (NatureCNN, sin CustomCNN)"
    return "CnnPolicy + CustomCNN"


def _make_ppo(args, venv):
    from stable_baselines3 import PPO
    kwargs = dict(n_steps=64, batch_size=64, n_epochs=1,
                  device=args.device, verbose=1)
    if args.mlp_policy:
        if not (args.synthetic_env or args.use_mock or args.init_order == "model-first"):
            raise ValueError("--mlp-policy solo con --synthetic-env o --use-mock.")
        policy = "MlpPolicy"
    elif args.no_custom_cnn:
        policy = "CnnPolicy"
    else:
        from src.cnn import CustomCNN
        policy = "CnnPolicy"
        kwargs["policy_kwargs"] = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=config.FEATURES_DIM))
    log("sb3 init", f"construyendo PPO [{_describe_policy(args)}] device={args.device}...")
    model = PPO(policy, venv, **kwargs)
    log("sb3 init", "PPO construido OK")
    return model


# --------------------------------------------------------------------------- #
# Modos
# --------------------------------------------------------------------------- #
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
            obs, reward, done, info = env.step(_sample_base_action(env))
            if i < 3 or i == args.steps - 1:
                log("step loop", f"step={i} reward={float(reward):.3f} done={bool(done)}")
            if done:
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
        assert obs.shape == (1, *config.STACKED_SHAPE), f"obs inesperada {obs.shape}"
        log("vec reset", f"obs {obs.shape}")
        log("step loop", f"ejecutando {args.steps} steps aleatorios...")
        for i in range(args.steps):
            obs, rewards, dones, infos = venv.step(np.array([venv.action_space.sample()]))
            if i < 3 or i == args.steps - 1:
                log("step loop", f"step={i} obs={obs.shape} reward={float(rewards[0]):.3f} done={bool(dones[0])}")
        log("done", "modo B completado sin crash")
    finally:
        venv.close()


def mode_check_spaces(args) -> None:
    log("init", f"check-spaces | synthetic={args.synthetic_env} | map={args.map}")
    venv = _build_sb3_env(args)
    try:
        os_ = venv.observation_space
        as_ = venv.action_space
        log("check-spaces", f"observation_space = {os_}")
        log("check-spaces", f"  shape={os_.shape} dtype={os_.dtype} "
                            f"low.min={np.min(os_.low)} high.max={np.max(os_.high)}")
        log("check-spaces", f"action_space = {as_}")
        log("check-spaces", f"  shape={as_.shape} dtype={getattr(as_,'dtype',None)} "
                            f"low={getattr(as_,'low',None)} high={getattr(as_,'high',None)}")
        obs = venv.reset()
        log("check-spaces", f"reset obs shape={obs.shape} dtype={obs.dtype}")
        log("done", "check-spaces completado (sin PPO)")
    finally:
        venv.close()


def _sb3_model_first(args, learn: bool) -> None:
    """model-first: PPO sobre un env sintético compatible, luego set_env(real)."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    log("init", f"model-first | learn={learn} | policy=[{_describe_policy(args)}]")
    log("vec reset", "construyendo PLACEHOLDER sintético (espacios del DuckieWrapper)...")
    placeholder = DummyVecEnv([lambda: SyntheticEnv(duckie_action=True)])
    model = _make_ppo(args, placeholder)  # PPO/torch se inicializan ANTES de tocar Duckietown
    placeholder.close()

    log("vec reset", "construyendo entorno REAL Duckietown...")
    real = _build_real_env(args)
    tmp = os.path.join("models", "_debug_ppo")
    tmp_zip = tmp + ".zip"
    try:
        log("sb3 init", "model.set_env(entorno real Duckietown)...")
        model.set_env(real)
        log("sb3 init", "set_env OK")
        if learn:
            log("sb3 learn", f"PPO.learn(total_timesteps={args.timesteps})...")
            model.learn(total_timesteps=args.timesteps)
            log("sb3 learn", "learn() terminó sin crash")
            os.makedirs("models", exist_ok=True)
            model.save(tmp)
        log("done", "model-first completado sin crash")
    finally:
        real.close()
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)


def mode_sb3_init(args) -> None:
    if args.init_order == "model-first":
        _sb3_model_first(args, learn=False)
        return
    log("init", f"modo C (sb3-init) | synthetic={args.synthetic_env} | "
                f"policy=[{_describe_policy(args)}]")
    venv = _build_sb3_env(args)
    try:
        _make_ppo(args, venv)
        log("done", "modo C completado: PPO inicializado sin entrenar")
    finally:
        venv.close()


def mode_sb3_learn(args) -> None:
    if args.init_order == "model-first":
        _sb3_model_first(args, learn=True)
        return
    log("init", f"modo D (sb3-learn) | synthetic={args.synthetic_env} | "
                f"timesteps={args.timesteps} | policy=[{_describe_policy(args)}]")
    venv = _build_sb3_env(args)
    tmp = os.path.join("models", "_debug_ppo")
    tmp_zip = tmp + ".zip"
    try:
        model = _make_ppo(args, venv)
        log("sb3 learn", f"PPO.learn(total_timesteps={args.timesteps})...")
        model.learn(total_timesteps=args.timesteps)
        log("sb3 learn", "learn() terminó sin crash")
        os.makedirs("models", exist_ok=True)
        model.save(tmp)
        log("done", "modo D completado; modelo temporal guardado y se borrará")
    finally:
        venv.close()
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)


def main(argv=None) -> None:
    args = parse_args(argv)
    if not args.synthetic_env:
        _reject_eval_map(args.map)
    print("=" * 64, **FLUSH)
    print(f"DEBUG DUCKIETOWN RUNTIME | gym_duckietown disponible: "
          f"{gym_duckietown_available()} | init-order={args.init_order}", **FLUSH)
    print("=" * 64, **FLUSH)

    if args.check_spaces:
        mode_check_spaces(args)
    elif args.sb3_learn:
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
