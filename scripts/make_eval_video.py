"""Genera un VÍDEO MP4 demostrativo del modelo final conduciendo en Duckietown.

Es una utilidad CUALITATIVA y OPCIONAL: la evaluación cuantitativa oficial sigue siendo
`eval.py`. Este script carga un modelo (.zip) entrenado con `train.py`, reconstruye el
entorno IDÉNTICO al de entrenamiento (`build_vec_env` + `VecFrameStack(4)` → obs
(4,64,64)) y graba el agente conduciendo, guardando un .mp4.

Por defecto graba UN ÚNICO rollout reproducible (con `--seed`, default 42). De forma
opcional, con `--rollouts > 1` corre varios rollouts y guarda el de MAYOR recompensa
acumulada (útil para descartar un rollout degenerado, p. ej. el coche parado).

No entrena nada. Reutiliza el patrón `model-first` de `eval.py` (cargar el modelo sobre
un entorno sintético y luego `set_env(real)`) para evitar el segfault de SB3 + Duckietown.

Ejemplos:
    # Local sin Duckietown (mock): valida el pipeline y produce un mp4 de ruido.
    python scripts/make_eval_video.py --use-mock --model models/best_agent \
        --out outputs/best_agent_loop_empty.mp4 --rollouts 2 --max-steps 100

    # Colab con Duckietown real (headless):
    xvfb-run -a python scripts/make_eval_video.py --model models/best_agent \
        --map Duckietown-loop_empty-v0 --out outputs/best_agent_loop_empty.mp4 --rollouts 1 --seed 42
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from stable_baselines3 import A2C, DQN, PPO, SAC
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from src import config
from src.envs import build_vec_env
from src.wrappers import SAFE_DISCRETE_ACTIONS

# ppo_adv / ppo_adv_v2 (Fase 3) cargan con la clase PPO (PPO con hiperparámetros avanzados).
ALGO_CLASSES = {"dqn": DQN, "ppo": PPO, "ppo_adv": PPO, "ppo_adv_v2": PPO,
                "sac": SAC, "a2c": A2C}


def set_global_seeds(seed: int) -> None:
    """Fija la semilla en random, numpy y torch (reproducibilidad razonable; NO promete
    resultados bit a bit, sobre todo en GPU o con el simulador Duckietown)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _PlaceholderEnv(gym.Env):
    """Entorno sintético (sin Duckietown) con los MISMOS espacios que el real tras
    FrameStack, para el flujo `model-first`: el modelo se carga sobre este placeholder
    ANTES de crear Duckietown, y luego se hace `set_env(real)`."""

    metadata = {"render_modes": []}

    def __init__(self, discrete: bool, n_stack: int, max_steps: int = 100,
                 action_mode: str = "wheels"):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(n_stack, 64, 64), dtype=np.uint8)
        if action_mode == "safe_discrete":
            self.action_space = spaces.Discrete(len(SAFE_DISCRETE_ACTIONS))
        elif discrete:
            self.action_space = spaces.Discrete(len(config.DISCRETE_ACTIONS))
        else:
            # Debe coincidir con DuckieWrapper según action_mode.
            low0 = 0.0 if action_mode == "v_omega" else -1.0
            self.action_space = spaces.Box(
                low=np.array([low0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32)
        self.max_steps = max_steps
        self._n = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._n = 0
        return self.observation_space.sample(), {}

    def step(self, action):
        self._n += 1
        return self.observation_space.sample(), 0.0, False, self._n >= self.max_steps, {}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Genera un vídeo MP4 del modelo final conduciendo en Duckietown.")
    p.add_argument("--algo", default="ppo",
                   choices=["dqn", "ppo", "ppo_adv", "ppo_adv_v2", "sac", "a2c"],
                   help="Algoritmo del modelo (elige la clase para .load). Default: ppo.")
    p.add_argument("--model", default="models/best_agent",
                   help="Ruta al .zip del modelo (con o sin extensión).")
    p.add_argument("--map", default="Duckietown-loop_empty-v0",
                   help="Mapa de TRAIN_MAPS, o el de evaluación oculto (solo --allow-eval).")
    p.add_argument("--out", default="outputs/best_agent_loop_empty.mp4",
                   help="Ruta de salida del .mp4 (se crea el directorio si falta).")
    p.add_argument("--rollouts", type=int, default=1,
                   help="Nº de rollouts. Default 1 = un único rollout reproducible (con "
                        "--seed). Solo si --rollouts > 1 se prueban varios y se guarda el "
                        "de MAYOR recompensa.")
    p.add_argument("--max-steps", type=int, default=1500,
                   help="Límite de pasos por rollout (corta rollouts que no terminan).")
    p.add_argument("--fps", type=int, default=30, help="FPS del vídeo de salida.")
    p.add_argument("--use-mock", action="store_true",
                   help="Forzar entorno mock (sin Duckietown; produce ruido, solo prueba).")
    p.add_argument("--allow-eval", action="store_true",
                   help="Habilita Duckietown-loop_obstacles-v0 SOLO PARA EVALUACIÓN.")
    p.add_argument("--action-mode", default="wheels",
                   choices=["wheels", "wheels_fixed", "v_omega", "v_omega_safe", "safe_discrete"],
                   help="Semántica de la acción; debe COINCIDIR con la usada al entrenar. "
                        "'wheels' = [left_wheel, right_wheel]; 'v_omega' = [v, omega] -> "
                        "ruedas; 'v_omega_safe' = v_omega acotado; 'safe_discrete' = "
                        "Discrete(5) de maniobras seguras.")
    p.add_argument("--reset-mode", default="default", choices=["default", "centerline"],
                   help="'default' = spawn aleatorio; 'centerline' = reset filtrado hasta "
                        "una pose inicial válida (drivable).")
    p.add_argument("--video-source", default="auto",
                   choices=["auto", "render_obs", "render"],
                   help="Fuente del frame de vídeo. 'auto' (default): prueba render_obs() "
                        "del simulador y luego render(). 'render_obs': SOLO render_obs() "
                        "(la cámara RGB del robot; falla si no existe). 'render': solo las "
                        "rutas render() (framebuffer, que en Xvfb puede salir ruidoso).")
    p.add_argument("--seed", type=int, default=42,
                   help="Semilla para random/numpy/torch y el entorno.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--n-stack", type=int, default=config.N_STACK)
    p.add_argument("--deterministic", dest="deterministic", action="store_true",
                   default=True, help="Política determinista (default).")
    p.add_argument("--stochastic", dest="deterministic", action="store_false",
                   help="Política estocástica.")
    return p.parse_args(argv)


# Se registra UNA vez por make_video() qué ruta de render se usó (diagnóstico).
_FRAME_SOURCE_LOGGED = {"done": False}


def _is_rgb(arr) -> bool:
    """True si arr parece un frame RGB(A): ndim==3 y 3 o 4 canales. Esto DESCARTA la
    observación del agente (1x64x64 / 4x64x64 o 64x64 gris), que no es un render válido."""
    arr = np.asarray(arr)
    return arr.ndim == 3 and arr.shape[2] in (3, 4)


def _coerce_rgb(arr) -> np.ndarray:
    """Convierte un frame válido a RGB uint8: RGBA -> RGB y dtype -> uint8."""
    arr = np.asarray(arr)
    if arr.shape[2] == 4:               # RGBA -> RGB
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        a = arr.astype(np.float64)
        if np.nanmax(a) <= 1.0 + 1e-6:  # floats en [0,1] -> [0,255]
            a = a * 255.0
        arr = np.clip(a, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _invoke(obj, method: str, kwargs: dict):
    """Llama obj.method(**kwargs) si existe. Devuelve un frame RGB(A) válido o None.

    Importante para Duckietown: `render_obs()` devuelve la CÁMARA RGB del robot (lo que ve
    el agente antes del preprocesado); `render()` devuelve el framebuffer de la ventana,
    que bajo Xvfb puede salir ruidoso aunque tenga shape RGB válida."""
    if obj is None:
        return None
    fn = getattr(obj, method, None)
    if fn is None or not callable(fn):
        return None
    try:
        frame = fn(**kwargs) if kwargs else fn()
    except Exception:
        return None
    if frame is not None and _is_rgb(frame):
        return frame
    return None


def get_rgb_frame(vec_env, last_obs=None, log: bool = False,
                  video_source: str = "auto") -> np.ndarray:
    """Obtiene SIEMPRE un frame RGB del SIMULADOR (no la observación del agente).

    Prioriza `render_obs()` (cámara RGB del robot) sobre `render()` (framebuffer). Con
    video_source='render_obs' solo se prueba render_obs y se falla si no existe; con
    'render' solo las rutas render(); 'auto' (default) prueba render_obs y luego render.
    Último recurso (solo auto/render): la observación del agente, con WARNING.
    """
    base = None
    try:
        base = vec_env.venv.envs[0]            # DuckieWrapper
    except Exception:
        base = None
    base_env = getattr(base, "env", None)      # gym_duckietown real (descendiendo por .env)
    base_unwrapped = getattr(base, "unwrapped", None)
    venv = getattr(vec_env, "venv", None)

    # (nombre, objeto, método, kwargs) — render_obs primero (cámara del robot).
    render_obs_sources = [
        ("venv.envs[0].env.render_obs", base_env, "render_obs", {}),
        ("venv.envs[0].unwrapped.render_obs", base_unwrapped, "render_obs", {}),
        ("venv.envs[0].render_obs", base, "render_obs", {}),
    ]
    render_sources = [
        ("venv.envs[0].env.render", base_env, "render", {"mode": "rgb_array"}),
        ("venv.envs[0].env.render", base_env, "render", {}),
        ("venv.envs[0].render", base, "render", {"mode": "rgb_array"}),
        ("venv.envs[0].render", base, "render", {}),
        ("venv.render", venv, "render", {"mode": "rgb_array"}),
        ("venv.render", venv, "render", {}),
        ("render", vec_env, "render", {"mode": "rgb_array"}),
        ("render", vec_env, "render", {}),
    ]
    if video_source == "render_obs":
        candidates = render_obs_sources
    elif video_source == "render":
        candidates = render_sources
    else:  # auto
        candidates = render_obs_sources + render_sources

    for name, obj, method, kwargs in candidates:
        frame = _invoke(obj, method, kwargs)
        if frame is not None:
            rgb = _coerce_rgb(frame)
            if log and not _FRAME_SOURCE_LOGGED["done"]:
                print(f"  [video] frame source={name} | shape={rgb.shape} | dtype={rgb.dtype}")
                _FRAME_SOURCE_LOGGED["done"] = True
            return rgb

    # video_source='render_obs': fallar explícitamente (no caer a render ni a obs).
    if video_source == "render_obs":
        raise RuntimeError(
            "--video-source render_obs: render_obs() no existe o falló en "
            "venv.envs[0].env / unwrapped / envs[0]. Usa --video-source auto o render.")

    # Fallback EXPLÍCITO a la observación (no es un render real): solo para no fallar.
    if last_obs is not None:
        obs = np.asarray(last_obs)
        gray = obs.reshape(-1, *obs.shape[-2:])[-1]   # último frame apilado (H, W)
        rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.uint8)
        if log and not _FRAME_SOURCE_LOGGED["done"]:
            print(f"  [video][WARNING] render del simulador no disponible; usando la "
                  f"OBSERVACIÓN del agente como frame (gris {rgb.shape}). El vídeo NO "
                  f"mostrará Duckietown real.")
            _FRAME_SOURCE_LOGGED["done"] = True
        return rgb

    raise RuntimeError(
        "No se pudo obtener un frame RGB del simulador (render_obs y render fallaron en "
        "todas las rutas) y no hay observación de respaldo.")


def _run_rollout(model, vec_env, max_steps: int, deterministic: bool,
                 video_source: str = "auto"):
    """Corre un rollout y devuelve (frames RGB, recompensa acumulada)."""
    frames = []
    obs = vec_env.reset()
    frames.append(get_rgb_frame(vec_env, last_obs=obs, log=True, video_source=video_source))
    total = 0.0
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, _ = vec_env.step(action)
        total += float(rewards[0])
        frames.append(get_rgb_frame(vec_env, last_obs=obs, video_source=video_source))
        if bool(dones[0]):
            break
    return frames, total


def _write_video(path: str, frames, fps: int) -> None:
    """Escribe los frames a un .mp4 con imageio (macro_block_size=1 admite dimensiones impares)."""
    import imageio
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, macro_block_size=1)


def make_video(args: argparse.Namespace) -> dict:
    discrete = args.algo == "dqn"
    cls = ALGO_CLASSES[args.algo]
    _FRAME_SOURCE_LOGGED["done"] = False  # re-log la fuente de frame en esta invocación

    # model-first: cargar el modelo sobre un env sintético y luego set_env(real).
    placeholder = DummyVecEnv([lambda: _PlaceholderEnv(
        discrete, args.n_stack, action_mode=args.action_mode)])
    model = cls.load(args.model, env=placeholder, device=args.device)
    vec_env = build_vec_env([args.map], discrete=discrete,
                            use_mock=(args.use_mock or None), seed=args.seed,
                            n_stack=args.n_stack, allow_eval=args.allow_eval,
                            action_mode=args.action_mode, reset_mode=args.reset_mode)
    model.set_env(vec_env)
    placeholder.close()

    best_frames, best_reward, best_idx = None, -np.inf, -1
    for i in range(args.rollouts):
        frames, reward = _run_rollout(model, vec_env, args.max_steps, args.deterministic,
                                      video_source=args.video_source)
        print(f"  rollout {i + 1}/{args.rollouts}: reward={reward:.3f} | frames={len(frames)}")
        if reward > best_reward:
            best_frames, best_reward, best_idx = frames, reward, i
    vec_env.close()

    _write_video(args.out, best_frames, args.fps)
    return {
        "out": args.out,
        "best_reward": float(best_reward),
        "best_rollout": int(best_idx) + 1,
        "frames": len(best_frames),
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    set_global_seeds(args.seed)
    print("=" * 64)
    print(f"VIDEO | algo={args.algo} | model={args.model} | map={args.map}")
    print(f"      | rollouts={args.rollouts} | seed={args.seed} | mock={args.use_mock} | "
          f"allow_eval={args.allow_eval} | out={args.out}")
    print("=" * 64)

    m = make_video(args)
    print("=" * 64)
    print(f"Mejor rollout      : #{m['best_rollout']} (recompensa {m['best_reward']:.3f})")
    print(f"Frames guardados   : {m['frames']}")
    print(f"Vídeo escrito en   : {m['out']}")
    print("=" * 64)


if __name__ == "__main__":
    main()
