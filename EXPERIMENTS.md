# EXPERIMENTS.md — Protocolo de entrenamiento real controlado (Duckietown RL)

Protocolo para entrenar y evaluar los agentes en **Duckietown real sobre Google Colab**,
registrar resultados y elegir el `best_agent.zip` final. El lanzador
`scripts/run_training_plan.py` compone los comandos correctos (no entrena solo).

> Entorno y fixes ya resueltos: stack del compañero (ver `COLAB_SETUP.md` §3-bis),
> Python 3.11 en venv, CPU forzada, y **`--init-order model-first`** (evita el segfault
> de SB3 + Duckietown real). `DuckieWrapper.step` normaliza la acción a `(2,)`.

## Precondiciones mínimas (una vez por runtime)
1. Runtime limpio; pasos 1-2 de `COLAB_SETUP.md` (clonar en `/content/MAML` + venv 3.11).
2. Instalar el stack del compañero (§3-bis).
3. Verificar imports (§4) y que `check_duckie_real.py` (§7) pasa.
4. **No** es necesario repetir todos los smoke tests en cada entrenamiento.

## Smoke real validado en Colab (baseline inicial de funcionamiento)

El ciclo completo está **validado** con el stack principal (§3-bis) y `--init-order
model-first`: Duckietown real → `train.py` → PPO learn → guardar → `eval.py` → evaluar
en mapa permitido y mapa oculto. Resultados del smoke (PPO corto real,
`models/ppo_colab_test.zip`):

| Mapa                         | Recompensa media | Longitud media |
|------------------------------|------------------|----------------|
| Duckietown-loop_empty-v0     | 1069.321         | 1500.0         |
| Duckietown-loop_obstacles-v0 | 600.820          | 1500.0         |

> ⚠️ Es un **baseline inicial de funcionamiento** (entrenamiento mínimo), **no el modelo
> final**. Sirve para confirmar que el pipeline end-to-end funciona; los modelos
> entregables salen de los entrenamientos controlados de abajo.

## Protocolo de entrenamiento controlado (orden recomendado)
a) En Colab, **`git pull origin main`** dentro de `/content/MAML` (traer el plan).
b) **Dry-run** de `ppo20k` para revisar el comando:
   `python scripts/run_training_plan.py --stage ppo20k --dry-run`
c) Ejecutar **`ppo20k`** con evaluación:
   `python scripts/run_training_plan.py --stage ppo20k --execute --eval-after`
d) Si pasa, ejecutar **`ppo50k`** (baseline PPO completo).
e) Después **DQN** (`dqn20k` → `dqn50k`) y **SAC** (`sac20k` → `sac50k`).

Registrar cada resultado en la tabla de abajo. Recordatorio:
**`Duckietown-loop_obstacles-v0` es SOLO evaluación, nunca entrenamiento.**

## Plan de entrenamiento real recomendado

> Nota de rendimiento: PPO 20k en **CPU** tardó ~**1h30**. Por eso conviene probar GPU
> con una run **corta** primero. El segfault de GPU era anterior a consolidar
> `model-first`; ahora merece la pena reintentarlo con una run de 5k, no con un
> entrenamiento largo. El lanzador **no reentrena** si el `.zip` ya existe (usa
> `--overwrite` para forzar).

**A. Pruebas cortas de estabilidad**
1. `ppo5k` en **CPU**: `python scripts/run_training_plan.py --stage ppo5k --execute --eval-after`
2. `ppo5k` en **GPU**: `... --stage ppo5k --execute --eval-after --use-gpu --device cuda`
3. Si GPU da `Segmentation fault`, **continuar en CPU** (no insistir con GPU).
4. Si GPU funciona, usar GPU para `ppo20k`/`ppo50k`.

**B. Entrenamientos principales** (en este orden)
1. `ppo20k` → 2. `ppo50k` → 3. `dqn20k` → 4. `sac20k` → 5. `dqn50k` y `sac50k` solo si hay tiempo.

**C. Evaluación**
- Siempre en `Duckietown-loop_empty-v0` y `Duckietown-small_loop-v0` (`--eval-after`).
- En `Duckietown-loop_obstacles-v0` solo con `--allow-eval-hidden`, y solo para comparar
  modelos **candidatos**. **Nunca entrenar** en `loop_obstacles`.

**D. Tabla de resultados (rellenar)**

| stage | device | timesteps | tiempo aprox | loop_empty rew | loop_empty len | small_loop rew | small_loop len | loop_obstacles rew | loop_obstacles len | observaciones |
|-------|--------|-----------|--------------|----------------|----------------|----------------|----------------|--------------------|--------------------|---------------|
| ppo5k  | CPU |  5 000 |        |  |  |  |  |  |  |  |
| ppo5k  | GPU |  5 000 |        |  |  |  |  |  |  |  |
| ppo20k |     | 20 000 |        |  |  |  |  |  |  |  |
| ppo50k |     | 50 000 |        |  |  |  |  |  |  |  |
| dqn20k |     | 20 000 |        |  |  |  |  |  |  |  |
| sac20k |     | 20 000 |        |  |  |  |  |  |  |  |
| dqn50k |     | 50 000 |        |  |  |  |  |  |  |  |
| sac50k |     | 50 000 |        |  |  |  |  |  |  |  |

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

## Criterio para elegir `best_agent.zip`
La nota depende de la **generalización al mapa oculto** `loop_obstacles`. Elegir el
modelo que maximice, **en orden de prioridad**:
1. Mayor **recompensa acumulada media en `loop_obstacles`** (con `--allow-eval-hidden`).
2. En empate, mayor **longitud media de episodio** (supervivencia sin salirse/chocar).
3. Robustez entre mapas: buen desempeño también en `loop_empty` y `small_loop`
   (no sobreajustado a un único mapa).

Cuando el ganador esté claro, copiarlo al nombre EXACTO del contrato (en Colab):
```bash
!cp models/<ganador>.zip models/best_agent.zip
```
y cerrar el `requirements.txt` definitivo (ver `COLAB_SETUP.md` §10) + dry-run del
contrato en un Colab limpio.

## Reglas de seguridad
- **NUNCA entrenar en `Duckietown-loop_obstacles-v0`** (descalificación). El lanzador y
  `train.py`/`src.envs.make_env` lo bloquean con `ValueError`.
- `loop_obstacles` solo se **evalúa**, y solo con `--allow-eval-hidden` (→ `--allow-eval`).
- Mantener `--init-order model-first` en Colab. GPU funcionó con una run controlada
  (`--use-gpu --device cuda`) tras consolidar model-first.

## Resultados finales (Colab, GPU)

Entrenamiento real en Colab con el stack validado (§3-bis) y `--init-order model-first`.
Evaluación con `eval.py` (model-first), 3 episodios por mapa salvo indicación.

| stage      | device | timesteps | loop_empty (rew ± std / len) | small_loop (rew ± std / len) | loop_obstacles (rew ± std / len) | conclusión |
|------------|--------|-----------|------------------------------|------------------------------|----------------------------------|------------|
| ppo20k_gpu | GPU    | 20 000    | 960.964 ± 605.380 / 1500.0   | 317.528 ± 703.254 / 1500.0   | 1118.216 ± 488.214 / 1500.0      | **GANADOR** |
| ppo50k_gpu | GPU    | 50 000    | −813.731 ± 210.545 / 1500.0  | −1238.875 ± 393.613 / 1500.0 | −1105.378 ± 760.645 / 1500.0     | descartado (recompensas negativas) |

**Modelo ganador: `ppo_loop_empty_20k_gpu`.** Mejor recompensa en `loop_obstacles`
(1118.216) y positiva en los tres mapas. **`ppo50k_gpu` NO se selecciona**: aunque
mantiene `length 1500.0`, obtiene **recompensas negativas en los tres mapas** (más
timesteps no mejoró; probable degradación de la política).

`loop_obstacles` se usó **solo para evaluación** con `--allow-eval-hidden`; **nunca**
para entrenamiento.

> **`best_agent.zip` = copia de `ppo_loop_empty_20k_gpu.zip`** (4.6 MB), generada en
> Colab con `cp models/ppo_loop_empty_20k_gpu.zip models/best_agent.zip` (nombre que pide
> la presentación; durante el desarrollo se usó `best_duckie_agent.zip`). Se conserva
> como **artefacto externo de entrega**, NO se versiona en el repo (ver `.gitignore`).
> Carga final verificada con `eval.py` en `loop_empty`, `episodes=1`, `device=cpu`,
> `init-order=model-first` → sin error, `length 1500.0`.
