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
import sys

# Permitir importar el paquete src/ ejecutando desde la raíz del proyecto.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.logger import configure

from src import config
from src.cnn import CustomCNN
from src.envs import build_vec_env

MODELS_DIR = "models"
DEFAULT_LOG_DIR = "logs"


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

    raise ValueError(f"Algoritmo desconocido: {algo!r}")


def resolve_maps(map_arg: str) -> list[str]:
    """'all' -> TRAIN_MAPS; un nombre exacto -> [nombre]. Bloquea EVAL_MAP."""
    if map_arg == "all":
        return list(config.TRAIN_MAPS)
    if map_arg == config.EVAL_MAP:
        raise ValueError(
            f"'{map_arg}' es el mapa de EVALUACIÓN oculto: prohibido entrenar en él."
        )
    if map_arg not in config.TRAIN_MAPS:
        raise ValueError(
            f"Mapa {map_arg!r} no está en TRAIN_MAPS. Válidos: {config.TRAIN_MAPS} o 'all'."
        )
    return [map_arg]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrenar DQN/PPO/SAC en Duckietown.")
    p.add_argument("--algo", required=True, choices=["dqn", "ppo", "sac"])
    p.add_argument("--map", default="all",
                   help="Nombre exacto de TRAIN_MAPS o 'all' (default).")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None,
                   help="Nombre del modelo (sin extensión). Default {algo}_duckie.")
    p.add_argument("--use-mock", action="store_true",
                   help="Forzar entorno mock (validación local sin Duckietown).")
    p.add_argument("--smoke", action="store_true",
                   help="Corrida mínima: timesteps≈512 y buffers diminutos.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--n-stack", type=int, default=config.N_STACK)
    p.add_argument("--features-dim", type=int, default=config.FEATURES_DIM)
    p.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    spec = get_algo_spec(args.algo, smoke=args.smoke)
    maps = resolve_maps(args.map)
    timesteps = 512 if args.smoke else args.timesteps
    output = args.output or f"{args.algo}_duckie"
    use_mock = True if args.use_mock else None  # None -> autodetecta en Colab

    print("=" * 64)
    print(f"TRAIN | algo={args.algo} | maps={maps} | timesteps={timesteps}")
    print(f"       | mock={args.use_mock} | smoke={args.smoke} | device={args.device}")
    print("=" * 64)

    # Entorno vectorizado con FrameStack -> (n_stack, 64, 64). GUARD incluido.
    env = build_vec_env(maps, discrete=spec["discrete"], use_mock=use_mock,
                        seed=args.seed, n_stack=args.n_stack)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=args.features_dim),
    )
    model = spec["cls"](
        "CnnPolicy", env, policy_kwargs=policy_kwargs, seed=args.seed,
        device=args.device, verbose=1, **spec["hyperparams"],
    )

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
