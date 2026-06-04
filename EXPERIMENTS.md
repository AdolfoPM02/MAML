# EXPERIMENTS.md — Protocolo de entrenamiento real controlado (Duckietown RL)

Protocolo para entrenar y evaluar los agentes en **Duckietown real sobre Google Colab**,
registrar resultados y elegir el `best_duckie_agent.zip` final. El lanzador
`scripts/run_training_plan.py` compone los comandos correctos (no entrena solo).

> Entorno y fixes ya resueltos: stack del compañero (ver `COLAB_SETUP.md` §3-bis),
> Python 3.11 en venv, CPU forzada, y **`--init-order model-first`** (evita el segfault
> de SB3 + Duckietown real). `DuckieWrapper.step` normaliza la acción a `(2,)`.

## Precondiciones mínimas (una vez por runtime)
1. Runtime limpio; pasos 1-2 de `COLAB_SETUP.md` (clonar en `/content/MAML` + venv 3.11).
2. Instalar el stack del compañero (§3-bis).
3. Verificar imports (§4) y que `check_duckie_real.py` (§7) pasa.
4. **No** es necesario repetir todos los smoke tests en cada entrenamiento.

## Comandos de entrenamiento (vía lanzador)
Siempre con `--device cpu`, `--init-order model-first`, `MPLBACKEND=Agg`,
`CUDA_VISIBLE_DEVICES=""`, `xvfb-run -a` (el lanzador los añade):

```bash
%cd /content/MAML
# Ver el comando antes de lanzar:
python scripts/run_training_plan.py --stage ppo20k --dry-run
# Entrenar y evaluar:
python scripts/run_training_plan.py --stage ppo20k --execute --eval-after
```

Stages disponibles:

| stage   | algo | timesteps | output                  |
|---------|------|-----------|-------------------------|
| ppo20k  | PPO  | 20 000    | `ppo_loop_empty_20k`    |
| ppo50k  | PPO  | 50 000    | `ppo_baseline_50k`      |
| dqn20k  | DQN  | 20 000    | `dqn_loop_empty_20k`    |
| dqn50k  | DQN  | 50 000    | `dqn_baseline_50k`      |
| sac20k  | SAC  | 20 000    | `sac_loop_empty_20k`    |
| sac50k  | SAC  | 50 000    | `sac_advanced_50k`      |

Orden sugerido: empezar por los `*20k` (rápidos) para validar curvas; luego los `*50k`.

## Comandos de evaluación
Con `--eval-after`, el lanzador evalúa en `Duckietown-loop_empty-v0` y
`Duckietown-small_loop-v0`. Para la prueba del contrato en el mapa oculto:

```bash
python scripts/run_training_plan.py --stage ppo50k --execute --eval-after --allow-eval-hidden
```
`--allow-eval-hidden` añade una evaluación en `Duckietown-loop_obstacles-v0` con
`--allow-eval` (SOLO evaluación).

## Registro de resultados
Rellenar tras cada experimento (recompensa media ± std y longitud media de episodio):

| stage  | output               | loop_empty (rew ± std / len) | small_loop (rew ± std / len) | loop_obstacles (rew / len) | notas |
|--------|----------------------|------------------------------|------------------------------|----------------------------|-------|
| ppo20k | ppo_loop_empty_20k   |                              |                              |                            |       |
| ppo50k | ppo_baseline_50k     |                              |                              |                            |       |
| dqn20k | dqn_loop_empty_20k   |                              |                              |                            |       |
| dqn50k | dqn_baseline_50k     |                              |                              |                            |       |
| sac20k | sac_loop_empty_20k   |                              |                              |                            |       |
| sac50k | sac_advanced_50k     |                              |                              |                            |       |

## Criterio para elegir `best_duckie_agent.zip`
La nota depende de la **generalización al mapa oculto** `loop_obstacles`. Elegir el
modelo que maximice, **en orden de prioridad**:
1. Mayor **recompensa acumulada media en `loop_obstacles`** (con `--allow-eval-hidden`).
2. En empate, mayor **longitud media de episodio** (supervivencia sin salirse/chocar).
3. Robustez entre mapas: buen desempeño también en `loop_empty` y `small_loop`
   (no sobreajustado a un único mapa).

Cuando el ganador esté claro, copiarlo al nombre EXACTO del contrato (en Colab):
```bash
!cp models/<ganador>.zip models/best_duckie_agent.zip
```
y cerrar el `requirements.txt` definitivo (ver `COLAB_SETUP.md` §10) + dry-run del
contrato en un Colab limpio.

## Reglas de seguridad
- **NUNCA entrenar en `Duckietown-loop_obstacles-v0`** (descalificación). El lanzador y
  `train.py`/`src.envs.make_env` lo bloquean con `ValueError`.
- `loop_obstacles` solo se **evalúa**, y solo con `--allow-eval-hidden` (→ `--allow-eval`).
- Mantener `--device cpu` e `--init-order model-first` en Colab mientras el segfault de
  GPU/env-first no esté resuelto.
