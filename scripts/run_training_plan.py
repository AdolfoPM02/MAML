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

# stage -> algoritmo, timesteps y nombre de salida estándar.
STAGES = {
    "ppo20k": dict(algo="ppo", timesteps=20_000, output="ppo_loop_empty_20k"),
    "ppo50k": dict(algo="ppo", timesteps=50_000, output="ppo_baseline_50k"),
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
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
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
    pre = 'env MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" '
    if args.xvfb:
        pre += "xvfb-run -a "
    return pre


def train_command(args: argparse.Namespace, stage: dict, output: str) -> str:
    return (f'{_prefix(args)}{args.python} train.py '
            f'--algo {stage["algo"]} --map {args.map} '
            f'--timesteps {stage["timesteps"]} --output {output} '
            f'--device {args.device} --init-order {args.init_order}')


def eval_commands(args: argparse.Namespace, stage: dict, output: str) -> list[str]:
    cmds = []
    for m in EVAL_MAPS:
        cmds.append(f'{_prefix(args)}{args.python} eval.py '
                    f'--algo {stage["algo"]} --model models/{output} --map {m} '
                    f'--episodes {args.episodes} --device {args.device} '
                    f'--init-order {args.init_order}')
    if args.allow_eval_hidden:
        cmds.append(f'{_prefix(args)}{args.python} eval.py '
                    f'--algo {stage["algo"]} --model models/{output} '
                    f'--map {config.EVAL_MAP} --episodes {args.episodes} --allow-eval '
                    f'--device {args.device} --init-order {args.init_order}')
    return cmds


def main(argv=None) -> None:
    args = parse_args(argv)
    stage = STAGES[args.stage]
    output = args.output or stage["output"]

    # SEGURIDAD: jamás entrenar en el mapa de evaluación oculto.
    if args.map == config.EVAL_MAP:
        raise ValueError(
            f"'{args.map}' es el mapa de EVALUACIÓN oculto: prohibido entrenar en él "
            f"(descalificación). Usa un mapa de TRAIN_MAPS; loop_obstacles solo se "
            f"evalúa con --allow-eval-hidden."
        )

    commands = [("TRAIN", train_command(args, stage, output))]
    if args.eval_after:
        for c in eval_commands(args, stage, output):
            commands.append(("EVAL", c))

    execute = args.execute and not args.dry_run

    print("=" * 70)
    print(f"PLAN | stage={args.stage} | algo={stage['algo']} | "
          f"timesteps={stage['timesteps']} | output={output}")
    print(f"     | map(train)={args.map} | eval_after={args.eval_after} | "
          f"allow_eval_hidden={args.allow_eval_hidden} | execute={execute}")
    print("=" * 70)
    for kind, cmd in commands:
        print(f"[{kind}] {cmd}")
    print("=" * 70)

    if not execute:
        print("(dry-run: no se ejecuta nada. Usa --execute para lanzar.)")
        return

    for kind, cmd in commands:
        print(f"\n>>> Ejecutando [{kind}]: {cmd}\n", flush=True)
        subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
