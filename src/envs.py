"""Construcción de entornos vectorizados con FrameStack (Fase 2).

- `make_env`       : crea un DuckieWrapper (continuo) o DuckieWrapper +
                     DiscreteActionWrapper (discreto, para DQN), con el GUARD que
                     impide entrenar en el mapa de evaluación oculto.
- `build_vec_env`  : envuelve en DummyVecEnv + VecFrameStack(n_stack=4), pasando de
                     observaciones (1, 64, 64) a (4, 64, 64).
"""

from __future__ import annotations

from typing import Callable

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from . import config
from .wrappers import DiscreteActionWrapper, DuckieWrapper


def make_env(map_name: str, discrete: bool = False,
             use_mock: bool | None = None, seed: int = 0,
             allow_eval: bool = False) -> Callable[[], object]:
    """Devuelve un *thunk* que construye un entorno envuelto.

    discrete=True  -> DuckieWrapper + DiscreteActionWrapper (DQN).
    discrete=False -> DuckieWrapper continuo (PPO / SAC / TD3).

    GUARD: si `map_name` es el mapa de evaluación oculto y `allow_eval` es False,
    lanza ValueError. El código de entrenamiento nunca pasa allow_eval=True, por lo
    que es imposible entrenar en Duckietown-loop_obstacles-v0.
    """
    if map_name == config.EVAL_MAP and not allow_eval:
        raise ValueError(
            f"'{map_name}' es el mapa de EVALUACIÓN oculto y no puede usarse para "
            f"entrenar (descalificación). Usa uno de TRAIN_MAPS o, solo para "
            f"evaluación, pasa allow_eval=True."
        )

    def _thunk():
        env = DuckieWrapper(env_name=map_name, use_mock=use_mock, seed=seed)
        if discrete:
            env = DiscreteActionWrapper(env)
        return env

    return _thunk


def build_vec_env(map_names: list[str], discrete: bool = False,
                  use_mock: bool | None = None, seed: int = 0,
                  n_stack: int = config.N_STACK,
                  allow_eval: bool = False) -> VecFrameStack:
    """Crea un VecEnv multi-mapa con FrameStack -> observación (4, 64, 64)."""
    thunks = [
        make_env(m, discrete=discrete, use_mock=use_mock, seed=seed + i,
                 allow_eval=allow_eval)
        for i, m in enumerate(map_names)
    ]
    venv = DummyVecEnv(thunks)
    return VecFrameStack(venv, n_stack=n_stack)
