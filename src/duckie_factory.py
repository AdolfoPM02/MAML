"""Capa de abstracción del entorno base de Duckietown.

Tanto Duckietown REAL (gym_duckietown, API de gym antiguo) como el entorno MOCK de
desarrollo exponen la MISMA interfaz mínima, de modo que los wrappers de arriba
(`DuckieWrapper`, `DiscreteActionWrapper`) funcionan sin cambios en ambos:

    reset()  -> obs                              (RGB uint8, sin info)
    step(a)  -> (obs, reward, done, info)        (4-tupla, estilo gym antiguo)
    render(mode='rgb_array') -> frame RGB
    close()

`gym_duckietown` NO se instala en local (conflicto numpy<=1.20 con Python 3.11).
En local usamos el MOCK; en Colab, `make_base_env` detecta gym_duckietown y usa el
entorno real automáticamente.
"""

from __future__ import annotations

import numpy as np

# Resolución típica de la cámara de Duckietown (alto, ancho, canales).
RAW_OBS_SHAPE: tuple[int, int, int] = (480, 640, 3)


class MockDuckietownEnv:
    """Entorno falso que imita la API de gym antiguo de Duckietown.

    Genera imágenes RGB pseudo-aleatorias (semilla reproducible) para validar el
    pipeline de observación, los wrappers y la CNN sin depender de Duckietown.
    """

    def __init__(self, env_name: str = "Mock-Duckietown-v0",
                 raw_shape: tuple[int, int, int] = RAW_OBS_SHAPE,
                 max_steps: int = 500, seed: int = 0):
        self.env_name = env_name
        self.raw_shape = raw_shape
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)
        self._step_count = 0

    def _frame(self) -> np.ndarray:
        return self._rng.integers(0, 256, size=self.raw_shape, dtype=np.uint8)

    def reset(self):
        self._step_count = 0
        return self._frame()

    def step(self, action):  # noqa: ARG002 - acción ignorada en el mock
        self._step_count += 1
        obs = self._frame()
        reward = float(self._rng.normal(0.0, 1.0))
        done = self._step_count >= self.max_steps
        info = {"mock": True, "step": self._step_count}
        return obs, reward, done, info

    def render(self, mode: str = "rgb_array"):  # noqa: ARG002
        return self._frame()

    def close(self):
        pass


def gym_duckietown_available() -> bool:
    """True si gym_duckietown se puede importar (entorno real disponible)."""
    try:
        import gym_duckietown  # noqa: F401
        return True
    except Exception:
        return False


def make_base_env(env_name: str, use_mock: bool | None = None,
                  seed: int = 0):
    """Crea el entorno base (real o mock) con interfaz de gym antiguo.

    use_mock=None -> autodetecta: usa real si gym_duckietown está disponible.
    use_mock=True -> fuerza el mock (desarrollo local).
    use_mock=False -> fuerza el real (fallará si gym_duckietown no está instalado).
    """
    if use_mock is None:
        use_mock = not gym_duckietown_available()

    if use_mock:
        return MockDuckietownEnv(env_name=env_name, seed=seed)

    import gym as old_gym  # gym antiguo, requerido por gym_duckietown
    return old_gym.make(env_name)
