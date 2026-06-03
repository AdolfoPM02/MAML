"""Smoke test de Fase 2 (entorno MOCK, sin entrenar).

Valida el pipeline y el contrato de evaluación sin instalar Duckietown ni entrenar:
  1. Importar librerías (torch, stable_baselines3, cv2, gymnasium, src.*).
  2. Crear entorno base mock.
  3. reset() en wrapper continuo y discreto.
  4. Comprobar shape (1, 64, 64) uint8.
  5. Un step().
  6. VecFrameStack -> (4, 64, 64).
  7. CustomCNN: forward (B, 4, 64, 64) -> (B, 256).
  8. Construir PPO (continuo) y DQN (discreto) con CustomCNN, SIN .learn().
  9. Verificar el GUARD de loop_obstacles.

Ejecutar:
    .venv-maml/bin/python scripts/smoke_test_phase2.py
"""

from __future__ import annotations

import os
import sys

# Permitir importar el paquete src/ ejecutando desde la raíz del proyecto.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 1. Importar librerías -------------------------------------------------------
import cv2  # noqa: F401
import numpy as np
import torch
import gymnasium  # noqa: F401
from stable_baselines3 import DQN, PPO

from src import config
from src.cnn import CustomCNN
from src.duckie_factory import make_base_env
from src.envs import build_vec_env, make_env
from src.wrappers import DiscreteActionWrapper, DuckieWrapper

USE_MOCK = True  # desarrollo local: forzar mock (Duckietown no está en .venv-maml)
PASS = "[OK]"


def check(cond: bool, msg: str) -> None:
    assert cond, f"[FALLO] {msg}"
    print(f"{PASS} {msg}")


def main() -> None:
    print("=" * 64)
    print("SMOKE TEST FASE 2 (mock, sin entrenamiento)")
    print("=" * 64)

    # 2. Entorno base mock ----------------------------------------------------
    base = make_base_env(config.TRAIN_MAPS[0], use_mock=USE_MOCK)
    raw = base.reset()
    check(raw.ndim == 3 and raw.shape[2] == 3, f"env base mock reset -> RGB {raw.shape}")

    # 3-5. Wrappers continuo y discreto: reset, shape, step -------------------
    for label, discrete in [("continuo (PPO/SAC)", False), ("discreto (DQN)", True)]:
        env = DuckieWrapper(config.TRAIN_MAPS[0], use_mock=USE_MOCK)
        if discrete:
            env = DiscreteActionWrapper(env)
        obs, info = env.reset()
        check(obs.shape == config.OBS_SHAPE and obs.dtype == np.uint8,
              f"{label}: reset obs {obs.shape} {obs.dtype}")
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info = env.step(action)
        check(obs2.shape == config.OBS_SHAPE,
              f"{label}: step obs {obs2.shape}, 5-tupla OK (r={reward:.2f})")
        env.close()

    # 6. VecFrameStack -> (4, 64, 64) -----------------------------------------
    vec_cont = build_vec_env(config.TRAIN_MAPS, discrete=False, use_mock=USE_MOCK)
    check(vec_cont.observation_space.shape == config.STACKED_SHAPE,
          f"VecFrameStack continuo obs_space {vec_cont.observation_space.shape}")
    o = vec_cont.reset()
    check(o.shape == (len(config.TRAIN_MAPS), *config.STACKED_SHAPE),
          f"VecFrameStack continuo reset {o.shape}")

    vec_disc = build_vec_env(config.TRAIN_MAPS, discrete=True, use_mock=USE_MOCK)
    check(vec_disc.observation_space.shape == config.STACKED_SHAPE,
          f"VecFrameStack discreto obs_space {vec_disc.observation_space.shape}")

    # 7. CustomCNN forward ----------------------------------------------------
    cnn = CustomCNN(vec_cont.observation_space, features_dim=config.FEATURES_DIM)
    dummy = torch.zeros(2, *config.STACKED_SHAPE)
    feats = cnn(dummy)
    check(tuple(feats.shape) == (2, config.FEATURES_DIM),
          f"CustomCNN forward {tuple(feats.shape)}")

    # 8. Construir PPO y DQN con CustomCNN, SIN entrenar -----------------------
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=config.FEATURES_DIM),
    )
    ppo = PPO("CnnPolicy", vec_cont, policy_kwargs=policy_kwargs,
              n_steps=64, device="cpu", verbose=0)
    check(ppo.policy is not None, "PPO construido con CustomCNN (continuo, sin learn)")

    dqn = DQN("CnnPolicy", vec_disc, policy_kwargs=policy_kwargs,
              buffer_size=1000, device="cpu", verbose=0)
    check(dqn.policy is not None, "DQN construido con CustomCNN (discreto, sin learn)")

    # 9. GUARD del mapa de evaluación oculto ----------------------------------
    try:
        make_env(config.EVAL_MAP)
        raise AssertionError("[FALLO] el guard de loop_obstacles no se activó")
    except ValueError:
        check(True, f"GUARD activo: make_env('{config.EVAL_MAP}') bloqueado")

    vec_cont.close()
    vec_disc.close()
    print("=" * 64)
    print("SMOKE TEST FASE 2: TODO OK (sin entrenamiento)")
    print("=" * 64)


if __name__ == "__main__":
    main()
