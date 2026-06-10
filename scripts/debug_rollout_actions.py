"""Auditoría de un rollout: ¿el modelo realmente ACTÚA y se aplica bien el action_mode?

Diagnóstico (NO entrena). Carga un modelo SB3 (PPO/A2C) o usa una política aleatoria /
una acción constante, ejecuta un rollout en el entorno REAL idéntico al de eval.py /
make_eval_video.py (mismo build_vec_env + VecFrameStack, mismo action_mode/reset_mode) e
imprime, paso a paso, la acción del modelo, la acción MAPEADA a ruedas (desde info), la
recompensa, done y la pose (cur_pos/cur_angle) si el simulador la expone.

Sirve para detectar bugs de pipeline (no solo de algoritmo): política colapsada (siempre
la misma acción), action_mode que no se aplica, modelo que no se carga, etc.

Ejemplos:
    # Modelo PPO safe_discrete
    xvfb-run -a python scripts/debug_rollout_actions.py --algo ppo \
        --model models/ppo_safe_discrete_loop_5k --map Duckietown-loop_empty-v0 \
        --action-mode safe_discrete --steps 200
    # Baseline aleatorio (sin modelo) para comparar
    xvfb-run -a python scripts/debug_rollout_actions.py --random-policy \
        --map Duckietown-loop_empty-v0 --action-mode safe_discrete --steps 200
    # Acción fija "recto lento" (id 0) en safe_discrete
    xvfb-run -a python scripts/debug_rollout_actions.py --constant-action 0 \
        --map Duckietown-loop_empty-v0 --action-mode safe_discrete --steps 200
    # Validación local sin Duckietown
    python scripts/debug_rollout_actions.py --random-policy --use-mock \
        --action-mode safe_discrete --steps 10
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from stable_baselines3 import A2C, PPO
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from src import config
from src.envs import build_vec_env
from src.wrappers import SAFE_DISCRETE_ACTIONS

ALGO_CLASSES = {"ppo": PPO, "a2c": A2C}


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _PlaceholderEnv(gym.Env):
    """Env sintético con el MISMO action_space que el real según action_mode, para el
    flujo model-first (cargar el modelo sin Duckietown y luego set_env(real))."""

    metadata = {"render_modes": []}

    def __init__(self, action_mode: str, n_stack: int, max_steps: int = 100):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(n_stack, 64, 64), dtype=np.uint8)
        if action_mode == "safe_discrete":
            self.action_space = spaces.Discrete(len(SAFE_DISCRETE_ACTIONS))
        else:
            low0 = 0.0 if action_mode == "v_omega" else -1.0
            self.action_space = spaces.Box(
                low=np.array([low0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)
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
        description="Audita un rollout: acciones del modelo, mapeo y pose (sin entrenar).")
    p.add_argument("--algo", default="ppo", choices=["ppo", "a2c"],
                   help="Clase para cargar el modelo (.load). Ignorado con --random-policy "
                        "o --constant-action.")
    p.add_argument("--model", default=None,
                   help="Ruta al .zip del modelo. Obligatorio salvo --random-policy / "
                        "--constant-action.")
    p.add_argument("--map", default="Duckietown-loop_empty-v0")
    p.add_argument("--action-mode", default="safe_discrete",
                   choices=["wheels", "wheels_fixed", "v_omega", "v_omega_safe",
                            "safe_discrete"])
    p.add_argument("--reset-mode", default="default", choices=["default", "centerline"])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--n-stack", type=int, default=config.N_STACK)
    p.add_argument("--use-mock", action="store_true")
    p.add_argument("--allow-eval", action="store_true",
                   help="Permite loop_obstacles SOLO para diagnóstico (nunca entrena).")
    p.add_argument("--deterministic", dest="deterministic", action="store_true",
                   default=True, help="Política determinista (default).")
    p.add_argument("--stochastic", dest="deterministic", action="store_false")
    p.add_argument("--random-policy", action="store_true",
                   help="No carga modelo: usa action_space.sample() (baseline aleatorio).")
    p.add_argument("--constant-action", type=int, default=None,
                   help="No carga modelo: repite SIEMPRE esta acción discreta (safe_discrete, "
                        "0..4). Sirve para medir el reward de ir recto/girar fijo.")
    p.add_argument("--print-every", type=int, default=1,
                   help="Imprimir detalle cada N pasos (default 1 = todos).")
    p.add_argument("--action-transform", default="identity",
                   choices=["identity", "negate", "swap", "swap_negate"],
                   help="SOLO DIAGNÓSTICO (acciones continuas wheels/v_omega/v_omega_safe): "
                        "transforma la acción ANTES de env.step para probar convenciones "
                        "de rueda. identity=[l,r]; negate=[-l,-r]; swap=[r,l]; "
                        "swap_negate=[-r,-l]. En safe_discrete solo se permite identity.")
    return p.parse_args(argv)


def _transform_action(act: np.ndarray, transform: str) -> np.ndarray:
    """Aplica una transformación a una acción CONTINUA batched (1, 2). Devuelve (1, 2)."""
    a = np.asarray(act[0], dtype=np.float32).reshape(-1)
    if transform == "negate":
        a = -a
    elif transform == "swap":
        a = a[::-1].copy()
    elif transform == "swap_negate":
        a = (-a[::-1]).copy()
    return np.array([a], dtype=np.float32)


def _base_env(vec_env):
    """DuckieWrapper subyacente: VecFrameStack -> venv (DummyVecEnv) -> envs[0]."""
    return vec_env.venv.envs[0]


def _pose(base):
    pos = getattr(base.env, "cur_pos", None)
    ang = getattr(base.env, "cur_angle", None)
    pos = None if pos is None else [round(float(x), 4) for x in np.asarray(pos).reshape(-1)]
    ang = None if ang is None else round(float(ang), 4)
    return pos, ang


def main(argv=None) -> None:
    args = parse_args(argv)
    set_global_seeds(args.seed)

    use_model = not (args.random_policy or args.constant_action is not None)
    if use_model and not args.model:
        raise SystemExit("Falta --model (o usa --random-policy / --constant-action).")
    if args.constant_action is not None and args.action_mode != "safe_discrete":
        print("[WARN] --constant-action está pensado para safe_discrete; "
              f"action_mode={args.action_mode}.")
    if args.action_transform != "identity" and args.action_mode == "safe_discrete":
        raise SystemExit(
            f"--action-transform={args.action_transform} no aplica a safe_discrete "
            f"(acción discreta). Usa identity, o un action_mode continuo "
            f"(wheels/v_omega/v_omega_safe).")

    print("=" * 72)
    mode = ("RANDOM" if args.random_policy else
            f"CONSTANT({args.constant_action})" if args.constant_action is not None else
            f"MODEL({args.algo}:{args.model})")
    print(f"DEBUG ROLLOUT | {mode} | map={args.map} | action_mode={args.action_mode} | "
          f"reset_mode={args.reset_mode}")
    print(f"             | steps={args.steps} | mock={args.use_mock} | "
          f"deterministic={args.deterministic} | seed={args.seed} | "
          f"action_transform={args.action_transform}")
    print("=" * 72)

    # Construcción del entorno IDÉNTICA a eval.py / make_eval_video.py.
    model = None
    if use_model:
        placeholder = DummyVecEnv(
            [lambda: _PlaceholderEnv(args.action_mode, args.n_stack)])
        model = ALGO_CLASSES[args.algo].load(args.model, env=placeholder, device=args.device)
        env = build_vec_env([args.map], discrete=False,
                            use_mock=(args.use_mock or None), seed=args.seed,
                            n_stack=args.n_stack, allow_eval=args.allow_eval,
                            action_mode=args.action_mode, reset_mode=args.reset_mode)
        model.set_env(env)
        placeholder.close()
    else:
        env = build_vec_env([args.map], discrete=False,
                            use_mock=(args.use_mock or None), seed=args.seed,
                            n_stack=args.n_stack, allow_eval=args.allow_eval,
                            action_mode=args.action_mode, reset_mode=args.reset_mode)

    base = _base_env(env)
    obs = env.reset()
    total_reward = 0.0
    frames = 0
    actions_taken: list = []

    for step in range(args.steps):
        if args.random_policy:
            act = np.array([env.action_space.sample()])
        elif args.constant_action is not None:
            act = np.array([args.constant_action])
        else:
            act, _ = model.predict(obs, deterministic=args.deterministic)

        # Acción ANTES de transformar (la que produce el modelo/random/constant).
        raw_before = np.asarray(act[0]).reshape(-1)
        is_discrete = args.action_mode == "safe_discrete"
        # Transformación de diagnóstico (solo acciones continuas; en discreto se bloquea).
        if not is_discrete and args.action_transform != "identity":
            act = _transform_action(act, args.action_transform)
        action_after = np.asarray(act[0]).reshape(-1)

        obs, rewards, dones, infos = env.step(act)
        info = infos[0] if infos else {}
        reward = float(rewards[0])
        done = bool(dones[0])
        total_reward += reward
        frames += 1

        # Para el histograma: id discreto si existe, si no la tupla (post-transform) enviada.
        if "discrete_action_id" in info:
            actions_taken.append(int(info["discrete_action_id"]))
        else:
            actions_taken.append(tuple(round(float(x), 3) for x in action_after))

        if step % args.print_every == 0:
            pos, ang = _pose(base)
            mapped = info.get("mapped_action")
            mapped = None if mapped is None else [round(float(x), 4)
                                                  for x in np.asarray(mapped).reshape(-1)]
            extra = {k: info[k] for k in ("terminated_still", "discrete_action_id")
                     if k in info}
            print(f"[{step:4d}] obs={tuple(obs.shape)} "
                  f"raw_before={[round(float(x), 4) for x in raw_before]} "
                  f"after={[round(float(x), 4) for x in action_after]} "
                  f"mapped={mapped} mode={info.get('action_mode')} r={reward:.3f} done={done} "
                  f"pos={pos} angle={ang} {extra if extra else ''}")
        if done:
            # DummyVecEnv auto-resetea; lo señalamos y seguimos hasta agotar --steps.
            print(f"        -- episodio terminado en step {step} (reward acumulado "
                  f"{total_reward:.3f}); el VecEnv se reinicia automáticamente --")

    env.close()

    # --- Resumen -----------------------------------------------------------
    print("=" * 72)
    print(f"total_reward = {total_reward:.3f}")
    print(f"frames       = {frames}")
    counter = Counter(actions_taken)
    unique = sorted(counter, key=lambda k: (str(type(k)), k))
    print(f"unique_actions ({len(unique)}):")
    if args.action_mode == "safe_discrete":
        names = ["recto_lento", "izq_suave", "der_suave", "izq_media", "der_media"]
        for i in range(len(SAFE_DISCRETE_ACTIONS)):
            n = counter.get(i, 0)
            bar = "#" * int(40 * n / max(frames, 1))
            print(f"  {i} {names[i]:11} {SAFE_DISCRETE_ACTIONS[i]}: {n:4d} {bar}")
    else:
        for a in unique:
            print(f"  {a}: {counter[a]}")
    print(f"first_20_actions = {actions_taken[:20]}")

    if (args.action_mode == "safe_discrete" and len(unique) == 1
            and args.constant_action is None and not args.random_policy):
        print("\n[!!!] POLICY COLLAPSED: el agente repite SIEMPRE la misma acción "
              f"({unique[0]}). Señal de política degenerada / bug, no de conducción.")
    print("=" * 72)


if __name__ == "__main__":
    main()
