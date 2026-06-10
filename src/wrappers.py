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

import warnings

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import config
from .duckie_factory import make_base_env

# Acciones DISCRETAS SEGURAS (velocidades de rueda [left, right]) para action_mode
# "safe_discrete": PPO elige un entero 0..4 y el wrapper lo traduce a estas ruedas.
# Suaves a propósito (PPO continuo no aprendía conducción estable ni siquiera acotado).
SAFE_DISCRETE_ACTIONS: list[list[float]] = [
    [0.18, 0.18],  # 0: recto lento
    [0.12, 0.20],  # 1: izquierda suave
    [0.20, 0.12],  # 2: derecha suave
    [0.08, 0.22],  # 3: izquierda media
    [0.22, 0.08],  # 4: derecha media
]


class DuckieWrapper(gym.Env):
    """Adapta Duckietown (real o mock) a Gymnasium con observación (1, 64, 64).

    Pipeline de visión: recorte del 50% superior (cielo) -> escala de grises ->
    resize 64x64 -> canal-primero (1, 64, 64) uint8.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env_name: str = config.TRAIN_MAPS[0],
                 use_mock: bool | None = None, seed: int = 0,
                 action_mode: str = "wheels",
                 reset_mode: str = "default", max_reset_attempts: int = 20):
        super().__init__()
        self.env_name = env_name
        self.env = make_base_env(env_name, use_mock=use_mock, seed=seed)

        # --- MODO DE RESET ---------------------------------------------------
        # "default": comportamiento original (spawn aleatorio del simulador).
        # "centerline": reset FILTRADO. Muchos episodios empezaban con pose/orientación
        # mala -> Invalid pose / not drivable inmediato. Aquí repetimos reset() hasta
        # obtener una pose VÁLIDA (drivable) según el propio simulador (env._valid_pose),
        # hasta max_reset_attempts; si ninguna vale, se avisa y se usa la última. En el
        # mock (sin pose) es un no-op. No fijamos coordenadas por mapa (no fiables); el
        # filtrado es reproducible dado el seed (primera pose válida de la secuencia RNG).
        if reset_mode not in ("default", "centerline"):
            raise ValueError(
                f"reset_mode debe ser 'default' o 'centerline'; recibido {reset_mode!r}.")
        self._reset_mode = reset_mode
        self._max_reset_attempts = int(max_reset_attempts)

        # --- SEMÁNTICA REAL DE LA ACCIÓN -------------------------------------
        # El simulador de Duckietown (gym_duckietown.DuckietownEnv.step) NO espera
        # [velocidad, giro], sino las velocidades de las DOS RUEDAS:
        #     action = [vel_rueda_izquierda, vel_rueda_derecha]   en [-1, 1]
        # El wrapper original pasaba la acción directa tratándola como [v, giro], por lo
        # que el simulador la interpretaba como ruedas → mapeo incorrecto (el robot no
        # avanzaba recto / derivaba). `action_mode` hace esta semántica explícita:
        #   - "wheels"  : la política produce directamente [left_wheel, right_wheel].
        #                 action_space = Box([-1,-1], [1,1]); se pasa recortada a [-1,1].
        #   - "v_omega" : la política produce [v, omega] (avance, giro); el wrapper lo
        #                 convierte a ruedas  left = v - omega, right = v + omega.
        #                 action_space = Box([0,-1], [1,1]) (avance no negativo).
        #   - "v_omega_safe" : como v_omega pero ACOTADO/SUAVE. La política da
        #                 [a_speed, a_turn] en [-1,1]; se mapea a una velocidad en un
        #                 rango pequeño [v_min, v_max] y un giro limitado a omega_max,
        #                 evitando acciones iniciales bruscas (v_omega salía de pista al
        #                 instante). action_space = Box([-1,-1], [1,1]).
        #   - "safe_discrete" : acción DISCRETA. action_space = Discrete(5); cada índice
        #                 se traduce a velocidades de rueda de SAFE_DISCRETE_ACTIONS.
        #                 PPO continuo no aprendía conducción estable; aquí PPO elige
        #                 entre 5 maniobras seguras predefinidas.
        #   - "wheels_fixed" : CORRIGE la convención de ruedas. El diagnóstico
        #                 (debug_rollout_actions.py) mostró que best_agent.zip conduce
        #                 bien solo con la transformación 'swap_negate'; es decir, la
        #                 convención real es la opuesta+intercambiada a 'wheels'. Mismo
        #                 action_space que wheels (Box([-1,-1],[1,1])), pero:
        #                     mapped = [-raw[1], -raw[0]]
        #                 (rueda izq = -raw_right, rueda der = -raw_left).
        if action_mode not in ("wheels", "wheels_fixed", "v_omega", "v_omega_safe",
                               "safe_discrete"):
            raise ValueError(
                f"action_mode debe ser 'wheels', 'wheels_fixed', 'v_omega', "
                f"'v_omega_safe' o 'safe_discrete'; recibido {action_mode!r}.")
        self._action_mode = action_mode

        # Parámetros del modo seguro continuo (suaves a propósito).
        self._v_min = 0.10
        self._v_max = 0.25
        self._omega_max = 0.30

        if action_mode == "safe_discrete":
            self.action_space = spaces.Discrete(len(SAFE_DISCRETE_ACTIONS))
        elif action_mode == "v_omega":
            self.action_space = spaces.Box(
                low=np.array([0.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:  # "wheels", "wheels_fixed" y "v_omega_safe": todos Box([-1,-1], [1,1])
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        # Observación: contrato (1, 64, 64) uint8.
        self.observation_space = spaces.Box(
            low=0, high=255, shape=config.OBS_SHAPE, dtype=np.uint8
        )

        # Último frame RGB ORIGINAL del simulador (antes de gris/resize/stack), guardado
        # solo para generar vídeo: en Colab/Xvfb render() devuelve framebuffer basura y
        # esta versión de gym_duckietown no tiene render_obs(), así que la fuente fiable
        # de imagen es la propia observación cruda del entorno. NO afecta a lo que ve el
        # agente (que sigue siendo la observación preprocesada).
        self.last_rgb_frame = None

    @staticmethod
    def _is_rgb_frame(obs) -> bool:
        """True si obs parece un frame RGB(A) crudo: ndarray, ndim==3, 3 o 4 canales y
        alto/ancho razonables (>=16). Esto distingue la observación CRUDA del simulador
        de la procesada (1,64,64)/gris."""
        if not isinstance(obs, np.ndarray):
            return False
        return (obs.ndim == 3 and obs.shape[2] in (3, 4)
                and obs.shape[0] >= 16 and obs.shape[1] >= 16)

    @staticmethod
    def _coerce_rgb_frame(obs) -> np.ndarray:
        """Convierte un frame válido a una COPIA RGB uint8: RGBA->RGB, float[0,1]->uint8."""
        arr = np.asarray(obs)
        if arr.shape[2] == 4:                 # RGBA -> RGB
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            a = arr.astype(np.float64)
            if np.nanmax(a) <= 1.0 + 1e-6:    # floats en [0,1] -> [0,255]
                a = a * 255.0
            arr = np.clip(a, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr).copy()

    def _store_rgb_frame(self, obs) -> bool:
        """Guarda self.last_rgb_frame si obs es un frame RGB válido. Si no, mantiene el
        último válido (no rompe). Devuelve True si se guardó uno nuevo."""
        if self._is_rgb_frame(obs):
            self.last_rgb_frame = self._coerce_rgb_frame(obs)
            return True
        return False

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

    def _pose_ok(self) -> bool:
        """True si la pose inicial es válida/drivable según el simulador. Si no podemos
        comprobarlo (mock, o el env no expone _valid_pose), devolvemos True para no romper."""
        valid_fn = getattr(self.env, "_valid_pose", None)
        pos = getattr(self.env, "cur_pos", None)
        ang = getattr(self.env, "cur_angle", None)
        if valid_fn is None or pos is None:
            return True
        try:
            return bool(valid_fn(pos, ang))
        except Exception:
            return True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()
        if self._reset_mode == "centerline" and getattr(self.env, "cur_pos", None) is not None:
            # Reset FILTRADO: repetir hasta una pose válida (drivable) o agotar intentos.
            attempts = 1
            while not self._pose_ok() and attempts < self._max_reset_attempts:
                obs = self.env.reset()
                attempts += 1
            if not self._pose_ok():
                warnings.warn(
                    f"reset_mode=centerline: sin pose válida en {self._max_reset_attempts} "
                    f"intentos ({self.env_name}); se usa la última.")
        self._store_rgb_frame(obs)  # frame RGB crudo para vídeo (no afecta a la obs)
        return self._process_obs(obs), {}

    def _map_action(self, action) -> tuple[np.ndarray, np.ndarray, dict]:
        """Aplana/valida la acción y la mapea a velocidades de rueda [left, right].

        SB3 / DummyVecEnv puede entregar la acción con una dimensión extra (p. ej.
        (1, 2) en vez de (2,)); aplanamos y exigimos exactamente 2 escalares. Devuelve
        (raw_action, mapped_action, extra), con raw/mapped vectores (2,) float32 y extra
        un dict opcional (v, omega) para diagnóstico:
          - "wheels"       : mapped = [clip(a0,-1,1), clip(a1,-1,1)]  (ya son ruedas).
          - "v_omega"      : v=clip(a0,0,1), omega=clip(a1,-1,1);
                             left=clip(v-omega,-1,1), right=clip(v+omega,-1,1).
          - "v_omega_safe" : a_speed,a_turn=clip(a,-1,1);
                             v = v_min + (a_speed+1)/2 * (v_max-v_min);  omega = omega_max*a_turn;
                             left=clip(v-omega,-1,1), right=clip(v+omega,-1,1).
          - "safe_discrete": action es un entero 0..4; mapped = SAFE_DISCRETE_ACTIONS[idx].
          - "wheels_fixed" : raw=clip(a,-1,1); mapped = [-raw[1], -raw[0]] (convención
                             corregida: equivale a la transformación 'swap_negate').
        """
        if self._action_mode == "safe_discrete":
            idx = int(np.asarray(action).reshape(-1)[0])
            if not (0 <= idx < len(SAFE_DISCRETE_ACTIONS)):
                raise ValueError(
                    f"safe_discrete espera un entero 0..{len(SAFE_DISCRETE_ACTIONS) - 1}; "
                    f"recibido {action!r}.")
            mapped = np.asarray(SAFE_DISCRETE_ACTIONS[idx], dtype=np.float32)
            return idx, mapped, {"discrete_action_id": idx}

        raw = np.asarray(action, dtype=np.float32).reshape(-1)
        if raw.shape[0] != 2:
            raise ValueError(
                f"La acción de Duckietown debe tener 2 valores; recibido shape "
                f"{np.asarray(action).shape} -> {raw.shape}"
            )
        extra: dict = {}
        if self._action_mode == "v_omega":
            v = float(np.clip(raw[0], 0.0, 1.0))
            omega = float(np.clip(raw[1], -1.0, 1.0))
            left = float(np.clip(v - omega, -1.0, 1.0))
            right = float(np.clip(v + omega, -1.0, 1.0))
        elif self._action_mode == "v_omega_safe":
            a_speed = float(np.clip(raw[0], -1.0, 1.0))
            a_turn = float(np.clip(raw[1], -1.0, 1.0))
            v = self._v_min + ((a_speed + 1.0) / 2.0) * (self._v_max - self._v_min)
            omega = self._omega_max * a_turn
            left = float(np.clip(v - omega, -1.0, 1.0))
            right = float(np.clip(v + omega, -1.0, 1.0))
            extra = {"v": float(v), "omega": float(omega)}
        elif self._action_mode == "wheels_fixed":
            a0 = float(np.clip(raw[0], -1.0, 1.0))
            a1 = float(np.clip(raw[1], -1.0, 1.0))
            raw = np.array([a0, a1], dtype=np.float32)  # raw_action = acción clipped
            left = -a1   # convención corregida (swap_negate)
            right = -a0
        else:  # "wheels"
            left = float(np.clip(raw[0], -1.0, 1.0))
            right = float(np.clip(raw[1], -1.0, 1.0))
        mapped = np.array([left, right], dtype=np.float32)
        return raw, mapped, extra

    def step(self, action):
        raw_action, mapped_action, extra = self._map_action(action)
        # El simulador recibe SIEMPRE velocidades de rueda [left, right].
        obs, reward, done, info = self.env.step(mapped_action)
        info["raw_action"] = raw_action
        info["mapped_action"] = mapped_action
        info["action_mode"] = self._action_mode
        info.update(extra)  # v / omega cuando aplica (v_omega_safe)
        if self._store_rgb_frame(obs):  # frame RGB crudo para vídeo (no afecta a la obs)
            info["has_last_rgb_frame"] = True
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
