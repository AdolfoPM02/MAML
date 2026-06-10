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

import warnings

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
    """True si el entorno REAL de Duckietown está disponible (no solo `import gym_duckietown`).

    Importar `gym_duckietown` (o sus submódulos) es lo que REGISTRA los entornos en gym;
    además se comprueba el registro de gym por si lo registrara otro paquete. NO instancia
    el simulador (eso es caro y puede segfaultear): solo verifica import + registro.
    """
    # 1) Importar alguno de los módulos que registran los entornos.
    for mod in ("gym_duckietown", "gym_duckietown.envs", "gym_duckietown.simulator"):
        try:
            __import__(mod)
            return True
        except Exception:
            continue
    # 2) ¿Hay algún entorno 'Duckietown-*' registrado en gym? (por si otro paquete lo hace)
    try:
        import gym as old_gym
        reg = old_gym.envs.registry
        keys = (reg.env_specs.keys() if hasattr(reg, "env_specs")
                else reg.keys() if hasattr(reg, "keys") else list(reg))
        if any("Duckietown" in str(k) for k in keys):
            return True
    except Exception:
        pass
    return False


def _related_env_ids(old_gym) -> list[str]:
    """IDs registrados en gym que parezcan de Duckietown (para mensajes de error útiles)."""
    try:
        reg = old_gym.envs.registry
        specs = (reg.values() if hasattr(reg, "values")
                 else reg.env_specs.values() if hasattr(reg, "env_specs") else [])
        ids = []
        for spec in specs:
            sid = getattr(spec, "id", str(spec))
            low = sid.lower()
            if "duckietown" in low or "duckie" in low or "loop" in low:
                ids.append(sid)
        return sorted(set(ids))
    except Exception:
        return []


def _make_real_env(env_name: str, seed: int):
    """Crea el entorno REAL con gym antiguo, o lanza RuntimeError claro si no se puede.

    IMPRESCINDIBLE: importar `gym_duckietown` ANTES de `gym.make`, porque ese import es lo
    que REGISTRA los IDs (p. ej. 'Duckietown-loop_empty-v0'). Sin él, gym.make da
    NameNotFound aunque el paquete esté instalado.
    """
    import gym as old_gym  # gym antiguo, requerido por gym_duckietown

    # 1) Registrar los entornos importando gym_duckietown.
    try:
        import gym_duckietown  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "No se pudo importar gym_duckietown (necesario para registrar los entornos "
            "reales). Instala duckietown-gym-daffy / gym-duckietown. Para pruebas SIN "
            "Duckietown usa use_mock=True (--use-mock); NO se cae al mock con use_mock=False."
        ) from e

    # 2) Crear el entorno; si falla, listar IDs registrados relacionados (ayuda mucho).
    try:
        env = old_gym.make(env_name)
    except Exception as e:
        related = _related_env_ids(old_gym)
        raise RuntimeError(
            f"No se pudo crear el entorno REAL de Duckietown '{env_name}': "
            f"{type(e).__name__}: {e}. IDs registrados relacionados "
            f"(Duckietown/duckie/loop): {related}. "
            f"Para pruebas SIN Duckietown usa use_mock=True (--use-mock)."
        ) from e

    print(f"[duckie_factory] Using real Duckietown env: {env_name}")
    # Best-effort: sembrar el entorno real. gym antiguo usa env.seed(); no rompemos si
    # no existe o falla (no se promete reproducibilidad bit a bit del simulador).
    try:
        if hasattr(env, "seed"):
            env.seed(seed)
    except Exception:
        pass
    return env


def make_base_env(env_name: str, use_mock: bool | None = None,
                  seed: int = 0):
    """Crea el entorno base (real o mock) con interfaz de gym antiguo.

    use_mock=True  -> fuerza el MOCK (desarrollo local). Avisa por consola.
    use_mock=False -> fuerza el REAL. Si no está disponible, lanza RuntimeError (NUNCA cae
                      al mock en silencio: así no se entrena/evalúa con ruido por accidente).
    use_mock=None  -> autodetecta: real si está disponible; si no, MOCK con WARNING explícito.
    """
    if use_mock is True:
        print(f"[duckie_factory] Using MOCK Duckietown env: {env_name}")
        return MockDuckietownEnv(env_name=env_name, seed=seed)

    if use_mock is False:
        # Real OBLIGATORIO: si falla, error claro (no mock silencioso).
        return _make_real_env(env_name, seed)

    # use_mock is None: autodetección.
    if gym_duckietown_available():
        return _make_real_env(env_name, seed)
    warnings.warn(
        f"Duckietown real no disponible (no se pudo importar gym_duckietown ni hay un "
        f"entorno 'Duckietown-*' registrado); usando MockDuckietownEnv para '{env_name}'. "
        f"Esto produce OBSERVACIONES DE RUIDO. Instala gym-duckietown para el entorno real, "
        f"o pasa use_mock=True (--use-mock) para silenciar este aviso en pruebas locales.",
        stacklevel=2,
    )
    print(f"[duckie_factory] Using MOCK Duckietown env: {env_name}")
    return MockDuckietownEnv(env_name=env_name, seed=seed)
