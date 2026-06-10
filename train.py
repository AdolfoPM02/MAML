"""Entrenamiento de agentes Duckietown (Fase 2B/3) con Stable-Baselines3.

Un único punto de entrada con CLI para entrenar:
  - DQN  : baseline con acción DISCRETA (DiscreteActionWrapper).
  - PPO  : baseline con acción CONTINUA.
  - SAC  : algoritmo avanzado (Fase 3) con acción CONTINUA.

Los tres usan `CustomCNN` sobre observaciones apiladas (4, 64, 64).

Local (sin Duckietown): usar `--use-mock` para validar el pipeline con el entorno
mock. En Colab (con gym_duckietown instalado) el MISMO comando, sin `--use-mock`,
entrena en Duckietown real: `make_base_env` autodetecta el entorno.

Ejemplos:
    # Validación local mínima (mock, entrenamiento corto)
    python train.py --algo ppo --use-mock --smoke --map Duckietown-loop_empty-v0 --output smoke_ppo
    # Entrenamiento real en Colab
    python train.py --algo sac --map all --timesteps 1000000 --output sac_duckie

NO entrena en Duckietown-loop_obstacles-v0 (mapa de evaluación oculto): bloqueado.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

# Permitir importar el paquete src/ ejecutando desde la raíz del proyecto.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import A2C, DQN, PPO, SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_schedule_fn

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from src import config
from src.cnn import CustomCNN
from src.envs import build_vec_env
from src.wrappers import SAFE_DISCRETE_ACTIONS

MODELS_DIR = "models"
DEFAULT_LOG_DIR = "logs"


def set_global_seeds(seed: int) -> None:
    """Fija la semilla en random, numpy y torch (reproducibilidad razonable; NO promete
    resultados bit a bit, sobre todo en GPU o con el simulador Duckietown)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _PlaceholderEnv(gym.Env):
    """Entorno sintético (sin Duckietown) con los MISMOS espacios que el entorno real
    tras FrameStack, usado por el flujo `model-first`: se construye el modelo SB3 sobre
    este placeholder ANTES de crear Duckietown, y luego se hace `set_env(real)`.

    obs    = Box(0, 255, (n_stack, 64, 64), uint8)
    action = Discrete(len(DISCRETE_ACTIONS))            si discrete (DQN)
             Box([-1,-1],[1,1], (2,), float32)          si continuo "wheels"
             Box([0,-1], [1,1], (2,), float32)          si continuo "v_omega"

    El espacio continuo DEBE coincidir con el de DuckieWrapper según action_mode; si no,
    set_env(real) fallaría por incompatibilidad de espacios.
    """

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


# --------------------------------------------------------------------------- #
# Especificación por algoritmo
# --------------------------------------------------------------------------- #
def get_algo_spec(algo: str, smoke: bool) -> dict:
    """Devuelve {cls, discrete, hyperparams} para el algoritmo pedido.

    Si `smoke` es True, reduce buffers / arranques para una corrida mínima.
    """
    if algo == "dqn":
        hp = dict(
            learning_rate=1e-4,
            buffer_size=50_000,
            learning_starts=10_000,
            batch_size=32,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1_000,
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            gamma=0.99,
        )
        if smoke:
            hp.update(buffer_size=2_000, learning_starts=100, target_update_interval=100)
        return dict(cls=DQN, discrete=True, hyperparams=hp)

    if algo == "ppo":
        hp = dict(
            learning_rate=3e-4,
            n_steps=2_048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
        )
        if smoke:
            hp.update(n_steps=128, batch_size=64, n_epochs=2)
        return dict(cls=PPO, discrete=False, hyperparams=hp)

    if algo == "ppo_adv":
        # Fase 3: PPO AVANZADO = PPO con HIPERPARÁMETROS diferenciados. Misma clase PPO,
        # mismo CustomCNN/wrappers y acción continua que el baseline; difiere en:
        #  - learning_rate menor (política más estable),
        #  - n_steps mayor (mejor estimación de ventaja / rollouts más largos),
        #  - ent_coef>0 (más exploración, evita colapso prematuro), vf_coef explícito.
        # NOTA: NO multimapa. Se descartó map=all porque rompe el flujo model-first
        # (set_env requiere mismo num_envs: 5 != 1). Se entrena en un solo mapa.
        hp = dict(
            learning_rate=1e-4,
            n_steps=4_096,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
        )
        if smoke:
            hp.update(n_steps=128, batch_size=64, n_epochs=2)
        return dict(cls=PPO, discrete=False, hyperparams=hp)

    if algo == "ppo_adv_v2":
        # Fase 3 (v2, CONSERVADORA): PPO casi idéntico al baseline ganador, con una mejora
        # real y pequeña: regularización/exploración por entropía suave (ent_coef=0.001).
        # ppo_adv (v1) degradó la política; v2 se mantiene cerca del baseline para no romper
        # lo que funciona. Misma clase PPO, CustomCNN, wrappers, model-first y acción continua.
        hp = dict(
            learning_rate=3e-4,
            n_steps=2_048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.001,
            vf_coef=0.5,
        )
        if smoke:
            hp.update(n_steps=128, batch_size=64, n_epochs=2)
        return dict(cls=PPO, discrete=False, hyperparams=hp)

    if algo == "sac":
        hp = dict(
            learning_rate=3e-4,
            buffer_size=50_000,
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
        )
        if smoke:
            hp.update(buffer_size=2_000, learning_starts=100, batch_size=64)
        return dict(cls=SAC, discrete=False, hyperparams=hp)

    if algo == "a2c":
        # A2C (RL puro on-policy, sugerido en las diapositivas). Más simple/rápido que
        # SAC/TD3. Funciona con espacio de acción Discrete (combina con safe_discrete) y
        # continuo. discrete=False: no añade DiscreteActionWrapper; con action_mode=
        # safe_discrete el propio DuckieWrapper ya expone Discrete(5).
        hp = dict(
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.0,
            vf_coef=0.5,
        )
        return dict(cls=A2C, discrete=False, hyperparams=hp)

    raise ValueError(f"Algoritmo desconocido: {algo!r}")


def resolve_maps(map_args: list[str] | str) -> list[str]:
    """Resuelve la lista de mapas de ENTRENAMIENTO (multi-mapa / curriculum).

    Acepta una lista (varios `--map`) o un solo nombre. 'all' se expande a TRAIN_MAPS.
    Valida cada mapa, BLOQUEA el mapa de evaluación oculto y elimina duplicados
    preservando el orden (relevante para curriculum: fácil -> difícil).
    """
    if isinstance(map_args, str):
        map_args = [map_args]

    expanded: list[str] = []
    for m in map_args:
        if m == "all":
            expanded.extend(config.TRAIN_MAPS)
        else:
            expanded.append(m)

    for m in expanded:
        if m == config.EVAL_MAP:
            raise ValueError(
                f"'{m}' es el mapa de EVALUACIÓN oculto: prohibido entrenar en él."
            )
        if m not in config.TRAIN_MAPS:
            raise ValueError(
                f"Mapa {m!r} no está en TRAIN_MAPS. Válidos: {config.TRAIN_MAPS} o 'all'."
            )
    # dedupe preservando orden
    return list(dict.fromkeys(expanded))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrenar DQN/PPO/SAC en Duckietown.")
    p.add_argument("--algo", required=True,
                   choices=["dqn", "ppo", "ppo_adv", "ppo_adv_v2", "sac", "a2c"])
    p.add_argument("--map", nargs="+", default=["all"],
                   help="Uno o VARIOS mapas de TRAIN_MAPS (multi-mapa / curriculum), o "
                        "'all' (default) para todos. Ej: --map Duckietown-straight_road-v0 "
                        "Duckietown-loop_empty-v0. Nunca loop_obstacles (bloqueado).")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=42,
                   help="Semilla para random/numpy/torch, el modelo SB3 y el entorno.")
    p.add_argument("--output", default=None,
                   help="Nombre del modelo (sin extensión). Default {algo}_duckie.")
    p.add_argument("--use-mock", action="store_true",
                   help="Forzar entorno mock (validación local sin Duckietown).")
    p.add_argument("--smoke", action="store_true",
                   help="Corrida mínima: timesteps≈512 y buffers diminutos.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--init-order", default="env-first",
                   choices=["env-first", "model-first"],
                   help="env-first (default): crea el env real y luego el modelo. "
                        "model-first: crea el modelo sobre un env sintético y luego "
                        "set_env(real) — evita el segfault de SB3 init con Duckietown.")
    p.add_argument("--n-stack", type=int, default=config.N_STACK)
    p.add_argument("--action-mode", default="wheels",
                   choices=["wheels", "wheels_fixed", "v_omega", "v_omega_safe", "safe_discrete"],
                   help="Semántica de la acción: 'wheels' (default) = [left_wheel, "
                        "right_wheel]; 'v_omega' = [v, omega] -> ruedas; 'v_omega_safe' = "
                        "v_omega con v/omega acotados; 'safe_discrete' = Discrete(5) de "
                        "maniobras seguras predefinidas (PPO discreto).")
    p.add_argument("--reset-mode", default="default", choices=["default", "centerline"],
                   help="'default' = spawn aleatorio del simulador; 'centerline' = reset "
                        "FILTRADO que repite hasta una pose inicial válida (drivable), "
                        "evitando empezar fuera de la carretera.")
    p.add_argument("--features-dim", type=int, default=config.FEATURES_DIM)
    p.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    p.add_argument("--init-model", default=None,
                   help="Ruta a un modelo guardado para CONTINUAR/fine-tune (PPO/ppo_adv*). "
                        "Si se da, se carga en vez de crear uno nuevo; la clase debe coincidir.")
    p.add_argument("--learning-rate-override", type=float, default=None,
                   help="Solo con --init-model: fuerza un learning_rate menor para "
                        "fine-tuning suave (sobrescribe lr y lr_schedule del modelo cargado).")
    return p.parse_args(argv)


def _build_model(spec, env, args, policy_kwargs):
    """Crea un modelo nuevo, o lo CARGA desde --init-model para continuar (fine-tune)."""
    if args.init_model:
        try:
            model = spec["cls"].load(args.init_model, env=env, device=args.device)
        except Exception as e:  # clase incompatible, fichero ausente, shapes, etc.
            raise ValueError(
                f"No se pudo cargar --init-model '{args.init_model}' como "
                f"{spec['cls'].__name__} (algo={args.algo}). ¿Coincide el algoritmo/modelo? "
                f"Detalle: {e}"
            )
        print(f"[fine-tune] modelo cargado desde {args.init_model} ({spec['cls'].__name__})")
        if args.learning_rate_override is not None:
            lr = args.learning_rate_override
            model.learning_rate = lr
            model.lr_schedule = get_schedule_fn(lr)  # SB3 usa lr_schedule en learn()
            print(f"[fine-tune] learning_rate_override={lr}")
        return model
    return spec["cls"](
        "CnnPolicy", env, policy_kwargs=policy_kwargs, seed=args.seed,
        device=args.device, verbose=1, **spec["hyperparams"],
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    set_global_seeds(args.seed)
    spec = get_algo_spec(args.algo, smoke=args.smoke)
    maps = resolve_maps(args.map)
    timesteps = 512 if args.smoke else args.timesteps
    output = args.output or f"{args.algo}_duckie"
    use_mock = True if args.use_mock else None  # None -> autodetecta en Colab

    print("=" * 64)
    print(f"TRAIN | algo={args.algo} | maps={maps} | timesteps={timesteps}")
    print(f"       | mock={args.use_mock} | smoke={args.smoke} | device={args.device} "
          f"| init-order={args.init_order} | seed={args.seed} | init-model={args.init_model} "
          f"| action_mode={args.action_mode} | reset_mode={args.reset_mode}")
    print("=" * 64)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=args.features_dim),
    )

    if args.init_order == "model-first":
        # Construir el modelo SB3 sobre un env SINTÉTICO (sin Duckietown) y luego
        # set_env(real). Evita el segfault de inicializar PPO/torch con Duckietown.
        # El placeholder debe tener TANTOS envs como mapas (multi-mapa): así num_envs
        # coincide con el VecEnv real y set_env(real) no falla (5 != 1).
        placeholder = DummyVecEnv(
            [(lambda: _PlaceholderEnv(spec["discrete"], args.n_stack,
                                      action_mode=args.action_mode))
             for _ in maps])
        model = _build_model(spec, placeholder, args, policy_kwargs)
        print(f"[model-first] modelo listo sobre {len(maps)} env(s) sintético(s); "
              f"creando Duckietown...")
        env = build_vec_env(maps, discrete=spec["discrete"], use_mock=use_mock,
                            seed=args.seed, n_stack=args.n_stack,
                            action_mode=args.action_mode, reset_mode=args.reset_mode)
        model.set_env(env)
        placeholder.close()
        print("[model-first] set_env(entorno real) OK")
    else:
        # env-first (default): crear el entorno real y luego el modelo.
        env = build_vec_env(maps, discrete=spec["discrete"], use_mock=use_mock,
                            seed=args.seed, n_stack=args.n_stack,
                            action_mode=args.action_mode, reset_mode=args.reset_mode)
        model = _build_model(spec, env, args, policy_kwargs)

    # Logger nativo SB3: stdout + CSV (sin dependencias nuevas).
    log_path = os.path.join(args.log_dir, args.algo)
    os.makedirs(log_path, exist_ok=True)
    model.set_logger(configure(log_path, ["stdout", "csv"]))

    model.learn(total_timesteps=timesteps)

    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, output)
    model.save(out_path)
    env.close()
    print(f"[OK] Modelo guardado en {out_path}.zip")


if __name__ == "__main__":
    main()
