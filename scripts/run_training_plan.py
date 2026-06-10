"""Lanzador/ayudante de experimentos de entrenamiento real en Colab (Duckietown).

NO entrena nada por defecto: compone los comandos de `train.py` (+ `eval.py` si se pide)
y solo los ejecuta con `--execute`. Con `--dry-run` (o sin `--execute`) únicamente los
imprime. Todos los comandos usan el patrón estable en Colab:

    env MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" xvfb-run -a {python} train.py ... \
        --device cpu --init-order model-first

Seguridad: NUNCA entrena en `Duckietown-loop_obstacles-v0` (mapa de evaluación oculto);
si se pasa como `--map` para entrenar, lanza ValueError. La evaluación en ese mapa solo
ocurre con `--allow-eval-hidden` (que añade `--allow-eval` a `eval.py`).

Ejemplos:
    python scripts/run_training_plan.py --stage ppo20k --dry-run
    python scripts/run_training_plan.py --stage ppo20k --execute --eval-after
    python scripts/run_training_plan.py --stage sac50k --execute --eval-after --allow-eval-hidden
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config

MODELS_DIR = "models"

# stage -> algoritmo, timesteps y nombre de salida estándar.
STAGES = {
    # Pruebas cortas de estabilidad (rápidas).
    "ppo5k": dict(algo="ppo", timesteps=5_000, output="ppo_loop_empty_5k"),
    "dqn5k": dict(algo="dqn", timesteps=5_000, output="dqn_loop_empty_5k"),
    "sac5k": dict(algo="sac", timesteps=5_000, output="sac_loop_empty_5k"),
    # Entrenamientos principales.
    "ppo20k": dict(algo="ppo", timesteps=20_000, output="ppo_loop_empty_20k"),
    "ppo50k": dict(algo="ppo", timesteps=50_000, output="ppo_baseline_50k"),
    # Rama experimental movement-reward-fix: PPO con el reward shaping de movimiento
    # del DuckieWrapper (premia desplazamiento real, penaliza quedarse parado). Mapa
    # por defecto loop_empty; single-map (model-first OK). NO sustituye a ppo20k.
    # min_forward_speed: fuerza avance mínimo (>0) para garantizar movimiento real y
    # evitar la política degenerada "parado" (velocidad clipada a 0).
    # eval_disable_shaping: el modelo se ENTRENA con shaping, pero se EVALÚA con la
    # recompensa LIMPIA del simulador (comparable con ppo20k y resultados previos).
    "ppo_move20k": dict(algo="ppo", timesteps=20_000, output="ppo_movement_20k",
                        min_forward_speed=0.1, eval_disable_shaping=True),
    "ppo_move50k": dict(algo="ppo", timesteps=50_000, output="ppo_movement_50k",
                        min_forward_speed=0.1, eval_disable_shaping=True),
    # Fase 3: PPO AVANZADO = PPO con HIPERPARÁMETROS diferenciados (algo=ppo_adv).
    # NO multimapa: se descartó map=all porque rompe --init-order model-first
    # (set_env num_envs 5 != 1). Usa el mapa por defecto (loop_empty), igual que ppo20k,
    # y admite override con --map (un mapa permitido); loop_obstacles sigue bloqueado.
    "ppo_adv5k":  dict(algo="ppo_adv", timesteps=5_000,  output="ppo_advanced_5k"),
    "ppo_adv20k": dict(algo="ppo_adv", timesteps=20_000, output="ppo_advanced_20k"),
    "ppo_adv50k": dict(algo="ppo_adv", timesteps=50_000, output="ppo_advanced_50k"),
    # Fase 3 v2 (CONSERVADORA): PPO ~baseline + entropía suave (ent_coef=0.001). Single-map.
    "ppo_adv_v2_5k":  dict(algo="ppo_adv_v2", timesteps=5_000,  output="ppo_advanced_v2_5k"),
    "ppo_adv_v2_20k": dict(algo="ppo_adv_v2", timesteps=20_000, output="ppo_advanced_v2_20k"),
    "ppo_adv_v2_50k": dict(algo="ppo_adv_v2", timesteps=50_000, output="ppo_advanced_v2_50k"),
    # Fase 3 (FINE-TUNING): continuar entrenando desde el modelo GANADOR (best_agent.zip).
    # Single-map (loop_empty); init-order model-first sigue funcionando (1 env == 1 env).
    "ppo_ft5k":  dict(algo="ppo", timesteps=5_000,  output="ppo_finetuned_5k",
                      init_model="models/best_agent", learning_rate_override=5e-5),
    "ppo_ft10k": dict(algo="ppo", timesteps=10_000, output="ppo_finetuned_10k",
                      init_model="models/best_agent", learning_rate_override=5e-5),
    "dqn20k": dict(algo="dqn", timesteps=20_000, output="dqn_loop_empty_20k"),
    "dqn50k": dict(algo="dqn", timesteps=50_000, output="dqn_baseline_50k"),
    "sac20k": dict(algo="sac", timesteps=20_000, output="sac_loop_empty_20k"),
    "sac50k": dict(algo="sac", timesteps=50_000, output="sac_advanced_50k"),
}

# Mapas de evaluación permitidos tras entrenar (nunca loop_obstacles salvo opt-in).
EVAL_MAPS = ["Duckietown-loop_empty-v0", "Duckietown-small_loop-v0"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lanzador de experimentos Duckietown.")
    p.add_argument("--stage", required=True, choices=sorted(STAGES.keys()))
    p.add_argument("--dry-run", action="store_true",
                   help="Imprime los comandos sin ejecutarlos.")
    p.add_argument("--execute", action="store_true",
                   help="Ejecuta el comando seleccionado (si no, solo imprime).")
    p.add_argument("--eval-after", action="store_true",
                   help="Tras entrenar, evalúa en los mapas definidos.")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=42,
                   help="Semilla (se pasa a train.py y eval.py). Default 42.")
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    p.add_argument("--use-gpu", action="store_true",
                   help="No fija CUDA_VISIBLE_DEVICES=\"\" (permite GPU). Por defecto se "
                        "fuerza CPU. Combinar con --device cuda (o auto).")
    p.add_argument("--overwrite", action="store_true",
                   help="Reentrenar aunque models/{output}.zip ya exista.")
    p.add_argument("--init-order", default="model-first",
                   choices=["env-first", "model-first"])
    p.add_argument("--python", default="/content/venv-maml/bin/python",
                   help="Intérprete del venv Python 3.11.")
    p.add_argument("--xvfb", action=argparse.BooleanOptionalAction, default=True,
                   help="Envolver con xvfb-run -a (default: sí).")
    p.add_argument("--map", default="Duckietown-loop_empty-v0",
                   help="Mapa de ENTRENAMIENTO (nunca loop_obstacles).")
    p.add_argument("--output", default=None,
                   help="Nombre de salida; si se omite, el estándar del stage.")
    p.add_argument("--allow-eval-hidden", action="store_true",
                   help="Permite EVALUAR (no entrenar) en Duckietown-loop_obstacles-v0.")
    return p.parse_args(argv)


def _prefix(args: argparse.Namespace) -> str:
    pre = "env MPLBACKEND=Agg "
    if not args.use_gpu:
        # Comportamiento por defecto: forzar CPU ocultando la GPU.
        pre += 'CUDA_VISIBLE_DEVICES="" '
    if args.xvfb:
        pre += "xvfb-run -a "
    return pre


def stage_train_map(args: argparse.Namespace, stage: dict) -> str:
    """Mapa de ENTRENAMIENTO efectivo. El mapa propio del stage PREVALECE sobre --map
    (p. ej. ppo_adv* fuerza map=all); si el stage no define mapa, se usa --map."""
    return stage.get("map", args.map)


def train_command(args: argparse.Namespace, stage: dict, output: str) -> str:
    cmd = (f'{_prefix(args)}{args.python} train.py '
           f'--algo {stage["algo"]} --map {stage_train_map(args, stage)} '
           f'--timesteps {stage["timesteps"]} --output {output} '
           f'--device {args.device} --init-order {args.init_order} --seed {args.seed}')
    if stage.get("init_model"):  # fine-tuning: continuar desde un modelo guardado
        cmd += f' --init-model {stage["init_model"]}'
        if stage.get("learning_rate_override") is not None:
            cmd += f' --learning-rate-override {stage["learning_rate_override"]}'
    if stage.get("min_forward_speed"):  # fuerza avance mínimo (p. ej. ppo_move*)
        cmd += f' --min-forward-speed {stage["min_forward_speed"]}'
    return cmd


def eval_commands(args: argparse.Namespace, stage: dict, output: str) -> list[str]:
    # Stages con shaping de movimiento (ppo_move*) se evalúan con recompensa LIMPIA.
    clean = " --disable-movement-shaping" if stage.get("eval_disable_shaping") else ""
    cmds = []
    for m in EVAL_MAPS:
        cmds.append(f'{_prefix(args)}{args.python} eval.py '
                    f'--algo {stage["algo"]} --model models/{output} --map {m} '
                    f'--episodes {args.episodes} --device {args.device} '
                    f'--init-order {args.init_order} --seed {args.seed}{clean}')
    if args.allow_eval_hidden:
        cmds.append(f'{_prefix(args)}{args.python} eval.py '
                    f'--algo {stage["algo"]} --model models/{output} '
                    f'--map {config.EVAL_MAP} --episodes {args.episodes} --allow-eval '
                    f'--device {args.device} --init-order {args.init_order} '
                    f'--seed {args.seed}{clean}')
    return cmds


def main(argv=None) -> None:
    args = parse_args(argv)
    stage = STAGES[args.stage]
    output = args.output or stage["output"]
    train_map = stage_train_map(args, stage)  # mapa efectivo (stage prevalece sobre --map)

    # SEGURIDAD: jamás entrenar en el mapa de evaluación oculto (se comprueba el EFECTIVO).
    if train_map == config.EVAL_MAP:
        raise ValueError(
            f"'{train_map}' es el mapa de EVALUACIÓN oculto: prohibido entrenar en él "
            f"(descalificación). Usa un mapa de TRAIN_MAPS o 'all'; loop_obstacles solo se "
            f"evalúa con --allow-eval-hidden."
        )

    # No reentrenar por accidente: si el .zip ya existe y no hay --overwrite, saltar
    # el entrenamiento (pero permitir evaluar si --eval-after).
    out_zip = os.path.join(MODELS_DIR, output + ".zip")
    skip_train = os.path.exists(out_zip) and not args.overwrite

    commands = []
    if not skip_train:
        commands.append(("TRAIN", train_command(args, stage, output)))
    if args.eval_after:
        for c in eval_commands(args, stage, output):
            commands.append(("EVAL", c))

    execute = args.execute and not args.dry_run
    gpu = "GPU" if args.use_gpu else "CPU"

    print("=" * 70)
    print(f"PLAN | stage={args.stage} | algo={stage['algo']} | "
          f"timesteps={stage['timesteps']} | output={output}")
    print(f"     | map(train)={train_map} | device={args.device} ({gpu}) | "
          f"eval_after={args.eval_after} | allow_eval_hidden={args.allow_eval_hidden} | "
          f"execute={execute}")
    print("=" * 70)
    if skip_train:
        print(f"[SKIP] Ya existe {out_zip}. Usa --overwrite para reentrenar.")
    for kind, cmd in commands:
        print(f"[{kind}] {cmd}")
    print("=" * 70)

    if not commands:
        print("(nada que hacer: modelo ya existe y no se pidió --eval-after.)")
        return
    if not execute:
        print("(dry-run: no se ejecuta nada. Usa --execute para lanzar.)")
        return

    for kind, cmd in commands:
        print(f"\n>>> Ejecutando [{kind}]: {cmd}\n", flush=True)
        subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
