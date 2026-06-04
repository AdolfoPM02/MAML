"""Prueba de Duckietown REAL en Colab (pasos 6-7 de la receta Colab).

Valida, contra el entorno real (o mock en local con --use-mock):
  - reset() del entorno base -> observación cruda RGB (480, 640, 3).
  - DuckieWrapper -> observación procesada (1, 64, 64) uint8.
  - build_vec_env + VecFrameStack(4) -> observación (4, 64, 64).

En Colab se ejecuta bajo display virtual, p. ej.:
    xvfb-run -a /content/venv-maml/bin/python scripts/check_duckie_real.py
    xvfb-run -a /content/venv-maml/bin/python scripts/check_duckie_real.py --reset-only

En local (sin Duckietown) solo para comprobar que el script es correcto:
    .venv-maml/bin/python scripts/check_duckie_real.py --use-mock
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import config
from src.duckie_factory import gym_duckietown_available, make_base_env
from src.envs import build_vec_env
from src.wrappers import DuckieWrapper

PASS = "[OK]"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probar Duckietown real: reset/wrappers/shapes.")
    p.add_argument("--map", default=config.TRAIN_MAPS[0],
                   help="Mapa de entrenamiento (nombre exacto de TRAIN_MAPS).")
    p.add_argument("--use-mock", action="store_true",
                   help="Forzar entorno mock (lint local sin Duckietown).")
    p.add_argument("--reset-only", action="store_true",
                   help="Solo probar reset() del entorno base (paso 6).")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    use_mock = True if args.use_mock else None  # None -> autodetecta gym_duckietown

    print("=" * 64)
    print(f"CHECK DUCKIETOWN | map={args.map} | use_mock={args.use_mock}")
    print(f"gym_duckietown disponible: {gym_duckietown_available()}")
    print("=" * 64)

    # Paso 6: reset() del entorno base -> RGB (480, 640, 3).
    base = make_base_env(args.map, use_mock=use_mock)
    raw = base.reset()
    raw = np.asarray(raw)
    assert raw.ndim == 3 and raw.shape[2] == 3, f"obs cruda inesperada: {raw.shape}"
    print(f"{PASS} reset() entorno base -> RGB {raw.shape}")
    base.close()

    if args.reset_only:
        print("=" * 64)
        print("RESET-ONLY OK")
        print("=" * 64)
        return

    # Paso 7a: DuckieWrapper -> (1, 64, 64) uint8.
    env = DuckieWrapper(args.map, use_mock=use_mock)
    obs, _ = env.reset()
    assert obs.shape == config.OBS_SHAPE and obs.dtype == np.uint8, \
        f"DuckieWrapper obs {obs.shape} {obs.dtype}"
    print(f"{PASS} DuckieWrapper -> {obs.shape} {obs.dtype}")
    env.close()

    # Paso 7b: build_vec_env + VecFrameStack(4) -> (4, 64, 64).
    vec = build_vec_env([args.map], discrete=False, use_mock=use_mock, n_stack=4)
    assert vec.observation_space.shape == config.STACKED_SHAPE, \
        f"VecFrameStack obs_space {vec.observation_space.shape}"
    o = vec.reset()
    assert o.shape == (1, *config.STACKED_SHAPE), f"VecFrameStack reset {o.shape}"
    print(f"{PASS} build_vec_env -> obs_space {vec.observation_space.shape}, reset {o.shape}")
    vec.close()

    print("=" * 64)
    print("CHECK DUCKIETOWN: TODO OK")
    print("=" * 64)


if __name__ == "__main__":
    main()
