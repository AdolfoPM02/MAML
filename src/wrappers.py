"""Wrappers de Duckietown -> Gymnasium (Fase 2).

- `DuckieWrapper`  : wrapper BASE común. Procesa la imagen a (1, 64, 64) uint8 y
                     hace de puente entre la API de gym antiguo del entorno base
                     y la API de Gymnasium (5-tupla) que espera Stable-Baselines3.
                     Expone un espacio de acción CONTINUO (PPO / SAC / TD3).
- `DiscreteActionWrapper` : capa fina ENCIMA de DuckieWrapper que expone un espacio
                     de acción DISCRETO (para DQN), mapeando cada índice a un comando
                     continuo. No altera la observación.

El mismo `DuckieWrapper` sirve para el entorno mock y para Duckietown real, porque
ambos comparten la interfaz definida en `duckie_factory.make_base_env`.
"""

from __future__ import annotations

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import config
from .duckie_factory import make_base_env


class DuckieWrapper(gym.Env):
    """Adapta Duckietown (real o mock) a Gymnasium con observación (1, 64, 64).

    Pipeline de visión: recorte del 50% superior (cielo) -> escala de grises ->
    resize 64x64 -> canal-primero (1, 64, 64) uint8.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env_name: str = config.TRAIN_MAPS[0],
                 use_mock: bool | None = None, seed: int = 0):
        super().__init__()
        self.env_name = env_name
        self.env = make_base_env(env_name, use_mock=use_mock, seed=seed)

        # Acción CONTINUA: [velocidad, giro] en [-1, 1] (PPO / SAC / TD3).
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        # Observación: contrato (1, 64, 64) uint8.
        self.observation_space = spaces.Box(
            low=0, high=255, shape=config.OBS_SHAPE, dtype=np.uint8
        )

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        # Recortar la mitad superior (cielo), pasar a grises y redimensionar.
        obs = obs[obs.shape[0] // 2:, :, :]
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (config.OBS_SHAPE[2], config.OBS_SHAPE[1]),
                             interpolation=cv2.INTER_AREA)
        processed = np.expand_dims(resized, axis=0).astype(np.uint8)  # (1, 64, 64)
        assert processed.shape == config.OBS_SHAPE, (
            f"Shape de observación {processed.shape} != {config.OBS_SHAPE}"
        )
        return processed

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()
        return self._process_obs(obs), {}

    def _normalize_action(self, action) -> np.ndarray:
        """Normaliza la acción a un vector PLANO de 2 escalares [velocidad, giro].

        SB3 / DummyVecEnv puede entregar la acción con una dimensión extra (p. ej.
        forma (1, 2) en vez de (2,)), lo que rompe la dinámica de Duckietown con
        `ValueError: setting an array element with a sequence`. Duckietown requiere
        exactamente dos escalares. Aplanamos, validamos y recortamos a rango válido:
        velocidad en [0, 1] y giro en [-1, 1].
        """
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != 2:
            raise ValueError(
                f"La acción de Duckietown debe tener 2 valores [velocidad, giro]; "
                f"recibido shape {np.asarray(action).shape} -> {a.shape}"
            )
        speed = float(np.clip(a[0], 0.0, 1.0))
        steering = float(np.clip(a[1], -1.0, 1.0))
        return np.array([speed, steering], dtype=np.float32)

    def step(self, action):
        action = self._normalize_action(action)
        obs, reward, done, info = self.env.step(action)
        # gym antiguo (4-tupla) -> gymnasium (5-tupla). El entorno base solo expone
        # 'done'; lo tratamos como 'terminated' y dejamos 'truncated' a la capa de
        # TimeLimit de SB3 si se usara. (En real Duckietown 'done' agrupa ambos.)
        return self._process_obs(obs), float(reward), bool(done), False, info

    def render(self):
        return self.env.render(mode="rgb_array")

    def close(self):
        self.env.close()


class DiscreteActionWrapper(gym.ActionWrapper):
    """Expone un espacio de acción DISCRETO para DQN sobre `DuckieWrapper`.

    Mapea cada índice discreto a un comando continuo [velocidad, giro] definido en
    `config.DISCRETE_ACTIONS`. La observación no se modifica.
    """

    def __init__(self, env: DuckieWrapper,
                 discrete_actions: list[list[float]] | None = None):
        super().__init__(env)
        self._actions = np.asarray(
            discrete_actions if discrete_actions is not None
            else config.DISCRETE_ACTIONS,
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(self._actions))

    def action(self, action: int) -> np.ndarray:
        return self._actions[int(action)]
