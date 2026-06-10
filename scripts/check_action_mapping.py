"""Diagnóstico del MAPEO DE ACCIONES de Duckietown (sin entrenar, sin cargar modelos).

Verifica empíricamente cómo se traduce la acción de la política a las velocidades de
rueda que espera el simulador, y si el robot se mueve como se espera. NO entrena ni carga
modelos: aplica acciones FIJAS durante unos pocos pasos y mide el desplazamiento real.

Recordatorio de semántica (ver src/wrappers.py):
  - El simulador espera  action = [vel_rueda_izquierda, vel_rueda_derecha]  en [-1, 1].
  - action_mode="wheels"  -> la política produce directamente [left, right].
  - action_mode="v_omega" -> la política produce [v, omega]; el wrapper convierte
                             left = v - omega, right = v + omega.

Casos probados por defecto (los pedidos en el plan):
  v_omega [0.2, 0.0] -> debería AVANZAR RECTO (left=right=0.2)
  v_omega [0.2, 0.2] -> debería GIRAR        (left=0.0, right=0.4)
  wheels  [0.2, 0.2] -> debería AVANZAR RECTO (ambas ruedas iguales)
  wheels  [0.1, 0.3] -> debería GIRAR        (ruedas distintas)

Local sin Duckietown: `--use-mock` (el mock IGNORA la acción y no tiene posición, así que
no habrá desplazamiento; solo valida el pipeline/mapeo). En Colab, con Duckietown real y
headless:

    xvfb-run -a python scripts/check_action_mapping.py --map Duckietown-loop_empty-v0
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import config
from src.wrappers import DuckieWrapper

# (action_mode, accion_fija, expectativa)
DEFAULT_CASES = [
    ("v_omega", [0.2, 0.0], "avanzar recto"),
    ("v_omega", [0.2, 0.2], "girar"),
    ("wheels",  [0.2, 0.2], "avanzar recto"),
    ("wheels",  [0.1, 0.3], "girar"),
    ("v_omega_safe", [-1.0, 0.0], "avanzar lento recto (v=v_min)"),
    ("v_omega_safe", [1.0, 0.0],  "avanzar algo más rápido recto (v=v_max)"),
    ("v_omega_safe", [0.0, 1.0],  "girar suave a un lado"),
    ("v_omega_safe", [0.0, -1.0], "girar suave al otro lado"),
]


def _get_pos(env: DuckieWrapper):
    """Posición del robot (vector) leyendo cur_pos del entorno base, o None si no existe
    (p. ej. el mock). Nunca lanza."""
    pos = getattr(env.env, "cur_pos", None)
    if pos is None:
        return None
    try:
        return np.asarray(pos, dtype=np.float64).reshape(-1)
    except Exception:
        return None


def run_case(action_mode: str, action, expectation: str, map_name: str,
             use_mock: bool, steps: int) -> dict:
    env = DuckieWrapper(env_name=map_name, use_mock=(use_mock or None),
                        action_mode=action_mode)
    env.reset()
    pos0 = _get_pos(env)
    mapped = None
    for _ in range(steps):
        _, _, terminated, truncated, info = env.step(action)
        mapped = info.get("mapped_action")
        if terminated or truncated:
            break
    pos1 = _get_pos(env)
    env.close()

    if pos0 is not None and pos1 is not None:
        dist = float(np.linalg.norm(pos1 - pos0))
    else:
        dist = None
    return {
        "action_mode": action_mode,
        "action": [round(float(x), 4) for x in np.asarray(action).reshape(-1)],
        "expectation": expectation,
        "mapped_action": None if mapped is None else
            [round(float(x), 4) for x in np.asarray(mapped).reshape(-1)],
        "pos0": None if pos0 is None else [round(float(x), 4) for x in pos0],
        "pos1": None if pos1 is None else [round(float(x), 4) for x in pos1],
        "distance": dist,
    }


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnóstico del mapeo de acciones (sin entrenar ni cargar modelos).")
    p.add_argument("--map", default="Duckietown-loop_empty-v0",
                   help="Mapa de TRAIN_MAPS (nunca el oculto; no se entrena).")
    p.add_argument("--use-mock", action="store_true",
                   help="Forzar mock (sin Duckietown; el mock ignora la acción).")
    p.add_argument("--steps", type=int, default=30,
                   help="Pasos por caso con la acción fija (default 30).")
    p.add_argument("--action-mode", default=None,
                   choices=["wheels", "v_omega", "v_omega_safe"],
                   help="Si se indica, solo se prueban los casos de ese modo; si se omite, "
                        "se prueban todos los casos por defecto (los tres modos).")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cases = DEFAULT_CASES
    if args.action_mode is not None:
        cases = [c for c in cases if c[0] == args.action_mode]

    print("=" * 72)
    print(f"CHECK ACTION MAPPING | map={args.map} | mock={args.use_mock} | "
          f"steps={args.steps}")
    print("=" * 72)
    for mode, action, expectation in cases:
        r = run_case(mode, action, expectation, args.map, args.use_mock, args.steps)
        print(f"[{mode:8}] accion={r['action']} -> ruedas(mapped)={r['mapped_action']} "
              f"| esperado: {expectation}")
        print(f"           pos0={r['pos0']} pos1={r['pos1']} | distancia="
              f"{'N/A (sin cur_pos)' if r['distance'] is None else round(r['distance'], 4)}")
    print("=" * 72)
    if args.use_mock:
        print("NOTA: el mock ignora la acción y no tiene cur_pos; la distancia es N/A. "
              "Ejecuta en Colab con Duckietown real (xvfb-run) para medir desplazamiento.")


if __name__ == "__main__":
    main()
