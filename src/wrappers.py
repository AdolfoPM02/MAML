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
                 use_mock: bool | None = None, seed: int = 0,
                 enable_movement_shaping: bool = True,
                 min_forward_speed: float = 0.0,
                 continuous_min_speed: float = 0.0,
                 movement_bonus: float = 5.0,
                 min_step_dist: float = 0.001,
                 still_step_penalty: float = 0.01,
                 max_still_steps: int = 150,
                 still_terminal_penalty: float = 100.0):
        super().__init__()
        self.env_name = env_name
        self.env = make_base_env(env_name, use_mock=use_mock, seed=seed)

        # Acción CONTINUA: [velocidad, giro]. La VELOCIDAD vive en [continuous_min_speed, 1]
        # (no negativa: marcha atrás dejaba al robot parado) y el GIRO en [-1, 1].
        # Con continuous_min_speed>0 (p. ej. 0.1) PPO ve directamente que la velocidad
        # MÍNIMA posible es positiva, así no puede converger a velocidad 0 (parado).
        self._continuous_min_speed = float(continuous_min_speed)
        self.action_space = spaces.Box(
            low=np.array([self._continuous_min_speed, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        # Observación: contrato (1, 64, 64) uint8.
        self.observation_space = spaces.Box(
            low=0, high=255, shape=config.OBS_SHAPE, dtype=np.uint8
        )

        # --- Reward shaping de MOVIMIENTO (rama experimental) -----------------
        # Premiar el desplazamiento real del robot y penalizar quedarse parado.
        # Solo se activa si el entorno base expone `cur_pos` (Duckietown real);
        # en el MOCK (sin cur_pos) el shaping es un no-op y la recompensa no cambia.
        # `enable_movement_shaping=False` evalúa con la recompensa LIMPIA del simulador
        # (comparable con resultados previos): no toca reward ni termina por quieto,
        # pero sigue rellenando info["step_dist"]/info["still_steps"] para diagnóstico.
        self._enable_movement_shaping = bool(enable_movement_shaping)
        # Velocidad mínima de avance forzada (0 = sin forzar). Con >0, la componente de
        # velocidad nunca baja de este valor, garantizando movimiento real durante el
        # entrenamiento (evita la política degenerada "parado"). Es una RESTRICCIÓN de
        # acción: una política entrenada con min_forward_speed>0 lo es bajo esa condición.
        self._min_forward_speed = float(min_forward_speed)
        self._movement_bonus = float(movement_bonus)
        self._min_step_dist = float(min_step_dist)
        self._still_step_penalty = float(still_step_penalty)
        self._max_still_steps = int(max_still_steps)
        self._still_terminal_penalty = float(still_terminal_penalty)
        self._last_pos = None
        self._still_steps = 0

    def _get_pos(self):
        """Posición actual del robot como vector float, o None si el entorno base no
        expone `cur_pos` (p. ej. el mock). Nunca lanza excepción."""
        pos = getattr(self.env, "cur_pos", None)
        if pos is None:
            return None
        try:
            return np.asarray(pos, dtype=np.float64).reshape(-1)
        except Exception:
            return None

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
        # Estado del shaping de movimiento: posición inicial y contador de "parado".
        self._last_pos = self._get_pos()
        self._still_steps = 0
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
        # Acción ORIGINAL (antes de recortar), para diagnóstico.
        raw_action = np.asarray(action, dtype=np.float32).reshape(-1)
        # Recortar a rango válido (velocidad [0,1], giro [-1,1]) y, si procede, forzar
        # una velocidad mínima de avance para garantizar movimiento real.
        clipped_action = self._normalize_action(action)
        if self._min_forward_speed > 0.0:
            clipped_action[0] = max(float(clipped_action[0]), self._min_forward_speed)
        obs, reward, done, info = self.env.step(clipped_action)
        reward = float(reward)
        done = bool(done)

        # --- Reward shaping de MOVIMIENTO -------------------------------------
        # Si hay posición disponible (Duckietown real), premiar el desplazamiento y
        # penalizar quedarse parado; si se acumulan demasiados pasos sin moverse,
        # terminar el episodio. En el mock (sin cur_pos) no se altera la recompensa.
        new_pos = self._get_pos()
        if new_pos is not None and self._last_pos is not None:
            step_dist = float(np.linalg.norm(new_pos - self._last_pos))
            # El contador de "parado" se actualiza SIEMPRE (sirve de diagnóstico
            # también con el shaping desactivado); solo la recompensa/terminación
            # dependen de enable_movement_shaping.
            if step_dist < self._min_step_dist:
                self._still_steps += 1
            else:
                self._still_steps = 0
            if self._enable_movement_shaping:
                reward += self._movement_bonus * step_dist
                if step_dist < self._min_step_dist:
                    reward -= self._still_step_penalty
                if self._still_steps >= self._max_still_steps:
                    done = True
                    reward -= self._still_terminal_penalty
                    info["terminated_still"] = True
        else:
            step_dist = None  # posición no disponible (mock): shaping desactivado
        if new_pos is not None:
            self._last_pos = new_pos
        info["step_dist"] = step_dist
        info["still_steps"] = self._still_steps
        info["raw_action"] = raw_action
        info["clipped_action"] = clipped_action

        # gym antiguo (4-tupla) -> gymnasium (5-tupla). El entorno base solo expone
        # 'done'; lo tratamos como 'terminated' y dejamos 'truncated' a la capa de
        # TimeLimit de SB3 si se usara. (En real Duckietown 'done' agrupa ambos.)
        return self._process_obs(obs), reward, done, False, info

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
