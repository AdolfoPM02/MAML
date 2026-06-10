"""Auditoría del PIPELINE DE OBSERVACIONES de Duckietown (sin entrenar, sin cargar modelos).

Captura la observación en CADA etapa y guarda PNGs + estadísticas, para localizar dónde
aparece el ruido: ¿ya viene ruidosa la cámara cruda del simulador (problema de render/GL
bajo Xvfb), o lo introduce el preprocesado del DuckieWrapper?

Etapas capturadas (en outputs/obs_debug/):
  a) raw_obs        : obs cruda del DuckietownEnv ANTES de DuckieWrapper (la cámara RGB).
  b) last_rgb_frame : copia RGB que guarda DuckieWrapper (debe coincidir con la cruda).
  c) wrapper_obs    : obs tras DuckieWrapper (gris 64x64), ANTES de VecFrameStack.
  d) vec_obs        : obs final tras VecFrameStack (4,64,64): cada canal + grid 2x2.

Para cada array imprime shape, dtype, min, max, mean, std, NaNs y una marca si parece
degenerado (std~0 = constante) o NaN.

Local sin Duckietown: `--use-mock` (la cámara del mock ES ruido aleatorio: sirve para
validar el script; en Colab real revela si la cámara verdadera es buena o ruido).

    xvfb-run -a python scripts/debug_observation_pipeline.py --map Duckietown-loop_empty-v0
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.duckie_factory import make_base_env
from src.wrappers import DuckieWrapper
from src.envs import build_vec_env


def _to_uint8(arr) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        return a
    a = a.astype(np.float64)
    if np.isfinite(a).any() and np.nanmax(a) <= 1.0 + 1e-6:
        a = a * 255.0
    return np.clip(np.nan_to_num(a), 0, 255).astype(np.uint8)


def _stats(name: str, arr) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    nan = int(np.isnan(a).sum())
    std = float(np.nanstd(a)) if a.size else 0.0
    flags = []
    if nan:
        flags.append("NaN!")
    if std < 1.0:
        flags.append("std~0 (CONSTANTE/degenerado)")
    # Heurística suave: gris uniforme aleatorio en [0,255] tiene std ~ 73.9.
    if 60.0 <= std <= 85.0 and abs(float(np.nanmean(a)) - 127.5) < 25.0:
        flags.append("posible RUIDO uniforme (std~74, media~128)")
    print(f"  {name:16} shape={tuple(np.shape(arr))} dtype={np.asarray(arr).dtype} "
          f"min={np.nanmin(a):.2f} max={np.nanmax(a):.2f} mean={np.nanmean(a):.2f} "
          f"std={std:.2f} nan={nan} {' '.join(flags)}")
    return {"std": std, "nan": nan, "mean": float(np.nanmean(a))}


def _save_rgb(path: str, arr) -> None:
    import imageio
    a = _to_uint8(arr)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[:, :, :3]
    imageio.imwrite(path, a)


def _save_gray(path: str, arr) -> None:
    import imageio
    g = _to_uint8(np.asarray(arr).squeeze())
    rgb = np.stack([g, g, g], axis=-1)
    imageio.imwrite(path, rgb)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audita el pipeline de observaciones (PNGs por etapa). No entrena.")
    p.add_argument("--map", default="Duckietown-loop_empty-v0")
    p.add_argument("--use-mock", action="store_true",
                   help="Mock (cámara aleatoria): solo valida el script.")
    p.add_argument("--steps", type=int, default=3, help="Steps tras el reset (1-5).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--action-mode", default="wheels_fixed",
                   choices=["wheels", "wheels_fixed", "v_omega", "v_omega_safe",
                            "safe_discrete"])
    p.add_argument("--out-dir", default="outputs/obs_debug")
    return p.parse_args(argv)


def _wrapper_action(action_mode):
    """Acción neutra válida para el wrapper según action_mode."""
    if action_mode == "safe_discrete":
        return 0
    return np.array([0.0, 0.0], dtype=np.float32)


def main(argv=None) -> None:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 72)
    print(f"DEBUG OBS PIPELINE | map={args.map} | mock={args.use_mock} | "
          f"steps={args.steps} | action_mode={args.action_mode} | out={args.out_dir}")
    print("=" * 72)

    # (a) Cámara CRUDA del DuckietownEnv (antes de cualquier wrapper).
    base = make_base_env(args.map, use_mock=args.use_mock, seed=args.seed)
    raw0 = base.reset()
    print("[a] raw_obs (cámara cruda del simulador, antes de DuckieWrapper):")
    _stats("raw_obs reset", raw0)
    _save_rgb(os.path.join(args.out_dir, "a_raw_obs_reset.png"), raw0)

    # (b)+(c) DuckieWrapper: last_rgb_frame (cruda guardada) + obs procesada (1,64,64).
    wrapper = DuckieWrapper(env_name=args.map, use_mock=args.use_mock,
                            seed=args.seed, action_mode=args.action_mode)
    proc0, _ = wrapper.reset()
    print("[b] last_rgb_frame (RGB crudo que guarda DuckieWrapper):")
    if wrapper.last_rgb_frame is not None:
        _stats("last_rgb_frame", wrapper.last_rgb_frame)
        _save_rgb(os.path.join(args.out_dir, "b_last_rgb_frame.png"), wrapper.last_rgb_frame)
    else:
        print("  (last_rgb_frame=None: la obs cruda no era RGB válido)")
    print("[c] wrapper_obs (tras DuckieWrapper, gris 64x64, antes de VecFrameStack):")
    _stats("wrapper_obs reset", proc0)
    _save_gray(os.path.join(args.out_dir, "c_wrapper_obs_reset.png"), proc0)

    # Unos pasos para ver variación temporal.
    act = _wrapper_action(args.action_mode)
    for i in range(1, args.steps + 1):
        proc, _, term, trunc, info = wrapper.step(act)
        _stats(f"wrapper_obs step{i}", proc)
        _save_gray(os.path.join(args.out_dir, f"c_wrapper_obs_step{i}.png"), proc)
        if term or trunc:
            print(f"  (episodio terminó en step {i})")
            break
    wrapper.close()

    # (d) Tras VecFrameStack: (1,4,64,64). Cada canal + grid 2x2.
    vec = build_vec_env([args.map], discrete=False,
                        use_mock=args.use_mock, seed=args.seed,
                        action_mode=args.action_mode)
    vobs = vec.reset()
    print("[d] vec_obs (tras VecFrameStack):")
    _stats("vec_obs", vobs)
    stack = np.asarray(vobs)[0]                 # (4,64,64)
    for c in range(stack.shape[0]):
        _save_gray(os.path.join(args.out_dir, f"d_vec_obs_ch{c}.png"), stack[c])
    # grid 2x2
    if stack.shape[0] == 4:
        row0 = np.concatenate([stack[0], stack[1]], axis=1)
        row1 = np.concatenate([stack[2], stack[3]], axis=1)
        grid = np.concatenate([row0, row1], axis=0)
        _save_gray(os.path.join(args.out_dir, "d_vec_obs_grid.png"), grid)
    vec.close()

    print("=" * 72)
    print(f"PNGs guardados en {args.out_dir}/ (a_*, b_*, c_*, d_*).")
    print("Interpretación: si [a] raw_obs ya es ruido (std~74, media~128) => el problema "
          "es el RENDER/GL del simulador (cámara), NO el preprocesado. Si [a] se ve bien "
          "pero [c]/[d] no, el bug estaría en DuckieWrapper._process_obs.")
    print("=" * 72)


if __name__ == "__main__":
    main()
