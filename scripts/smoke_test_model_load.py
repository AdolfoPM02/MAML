"""Smoke test del CONTRATO DE CARGA (mock, sin Duckietown).

Replica en local el punto crítico del contrato de evaluación: entrenar un modelo
mínimo, guardarlo, RECARGARLO sin pasar `custom_objects` (lo que obliga a que
`CustomCNN` y los wrappers se importen solos) y hacer `predict()`. Si esto pasa en
mock, el modelo "cargará a la primera" en la máquina del profesor.

Ejecutar:
    .venv-maml/bin/python scripts/smoke_test_model_load.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN, PPO, SAC

from src import config
from src.cnn import CustomCNN
from src.envs import build_vec_env

ALGO_CLASSES = {"dqn": DQN, "ppo": PPO, "sac": SAC}
MAP = config.TRAIN_MAPS[0]  # nombre exacto de TRAIN_MAPS
PASS = "[OK]"


def _train_save_load_predict(algo: str) -> None:
    cls = ALGO_CLASSES[algo]
    discrete = algo == "dqn"
    tmp_path = os.path.join("models", f"_loadtest_{algo}")  # SB3 añade .zip
    tmp_zip = tmp_path + ".zip"

    try:
        # 1. Entrenamiento mínimo con CustomCNN sobre entorno mock.
        env = build_vec_env([MAP], discrete=discrete, use_mock=True, n_stack=4)
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=config.FEATURES_DIM),
        )
        common = dict(device="cpu", seed=0, verbose=0)
        if discrete:
            model = cls("CnnPolicy", env, policy_kwargs=policy_kwargs,
                        buffer_size=2_000, learning_starts=100, **common)
        elif algo == "sac":
            model = cls("CnnPolicy", env, policy_kwargs=policy_kwargs,
                        buffer_size=2_000, learning_starts=100, batch_size=64, **common)
        else:  # ppo
            model = cls("CnnPolicy", env, policy_kwargs=policy_kwargs,
                        n_steps=128, batch_size=64, n_epochs=1, **common)
        model.learn(total_timesteps=256)

        # 2. Guardar.
        os.makedirs("models", exist_ok=True)
        model.save(tmp_path)
        env.close()
        del model
        assert os.path.exists(tmp_zip), f"no se guardó {tmp_zip}"

        # 3. RECARGAR sin custom_objects (CustomCNN debe importarse solo).
        eval_env = build_vec_env([MAP], discrete=discrete, use_mock=True, n_stack=4)
        loaded = cls.load(tmp_path, env=eval_env, device="cpu")

        # 4. predict() y validación del espacio de acción.
        obs = eval_env.reset()
        action, _ = loaded.predict(obs, deterministic=True)
        space = eval_env.action_space
        if isinstance(space, spaces.Discrete):
            ok = np.asarray(action).shape == (1,) and 0 <= int(action[0]) < space.n
        else:
            ok = np.asarray(action).shape == (1, *space.shape)
        eval_env.close()
        assert ok, f"acción {np.asarray(action).shape} no encaja con {space}"
        print(f"{PASS} {algo.upper()}: train -> save -> load(sin custom_objects) -> "
              f"predict OK (acción {np.asarray(action).tolist()})")
    finally:
        # 5. Borrado garantizado del temporal, aunque algo falle.
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)


def main() -> None:
    print("=" * 64)
    print("SMOKE TEST CONTRATO DE CARGA (mock)")
    print("=" * 64)
    for algo in ("ppo", "dqn", "sac"):
        _train_save_load_predict(algo)
    # Confirmar que no quedan temporales _loadtest_*.
    leftovers = [f for f in os.listdir("models") if f.startswith("_loadtest_")] \
        if os.path.isdir("models") else []
    assert not leftovers, f"quedaron temporales: {leftovers}"
    print(f"{PASS} sin temporales _loadtest_* restantes")
    print("=" * 64)
    print("CONTRATO DE CARGA: TODO OK")
    print("=" * 64)


if __name__ == "__main__":
    main()
