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
Evaluación con `eval.py` (model-first), 3 episodios por mapa. Formato: `recompensa media
± std / longitud media`.

| algoritmo / stage | output | timesteps | loop_empty | small_loop | loop_obstacles | decisión |
|---|---|---|---|---|---|---|
| DQN 20k | `dqn_loop_empty_20k` | 20 000 | −1013.352 ± 27.031 / 112.3 | −1001.872 ± 46.784 / 86.3 | −1065.766 ± 20.149 / 61.0 | descartado |
| DQN 50k | `dqn_baseline_50k` | 50 000 | −1047.611 ± 76.916 / 82.0 | −1016.568 ± 65.748 / 74.3 | −1032.042 ± 52.425 / 36.7 | descartado |
| **PPO 20k** | `ppo_loop_empty_20k_gpu` | 20 000 | 960.964 ± 605.380 / 1500.0 | 317.528 ± 703.254 / 1500.0 | **1118.216 ± 488.214 / 1500.0** | **GANADOR** |
| PPO 50k | `ppo_loop_empty_50k_gpu` | 50 000 | −813.731 ± 210.545 / 1500.0 | −1238.875 ± 393.613 / 1500.0 | −1105.378 ± 760.645 / 1500.0 | descartado |
| SAC 20k | `sac_loop_empty_20k` | 20 000 | 990.616 ± 139.264 / 1500.0 | −1253.567 ± 120.299 / 295.7 | 915.236 ± 571.946 / 1500.0 | evaluado, no supera a PPO 20k |
| SAC 50k | `sac_advanced_50k` | 50 000 (objetivo) | — | — | — | **no completado** (ver nota) |

**Modelo ganador: `ppo_loop_empty_20k_gpu`** (entregado como `best_agent.zip`). Es el de
mejor equilibrio global y la **mejor recompensa en el mapa oculto `loop_obstacles`**
(1118.216), por encima de SAC 20k (915.236) y de todo DQN. Lecturas por algoritmo:
- **DQN (20k y 50k):** baseline discreto **descartado** — recompensas negativas y
  **episodios muy cortos** (longitudes 36–112 ≪ 1500), el agente falla pronto;
  entrenar más (50k) no mejora.
- **PPO:** 20k es el ganador; **PPO 50k se descartó** (recompensas negativas: más
  entrenamiento degradó la política).
- **SAC 20k:** algoritmo avanzado **implementado, entrenado y evaluado**; buen
  `loop_empty` (990.616) y `loop_obstacles` (915.236) pero **mala generalización en
  `small_loop`** (negativa, longitud 295.7) y **no supera a PPO 20k** en el mapa oculto.

> **Nota — SAC 50k (no completado):** la ejecución se **interrumpió en Colab alrededor de
> 30 000 timesteps tras más de 6 h** y **no se generó el artefacto `sac_advanced_50k.zip`**.
> Por reproducibilidad y trazabilidad, **no se incluye como métrica cuantitativa válida**;
> figura solo como intento no completado.

`loop_obstacles` se usó **solo para evaluación** con `--allow-eval-hidden`; **nunca**
para entrenamiento (bloqueado por `ValueError` en el código).

> **`best_agent.zip` = copia de `ppo_loop_empty_20k_gpu.zip`** (4.6 MB), generada en
> Colab con `cp models/ppo_loop_empty_20k_gpu.zip models/best_agent.zip` (nombre que pide
> la presentación; durante el desarrollo se usó `best_duckie_agent.zip`). **Se incluye en
> la raíz de la entrega final** (ZIP autosuficiente) y **se versiona en el repo** vía la
> excepción `!best_agent.zip` del `.gitignore`; el notebook lo copia a `models/`
> automáticamente. Carga final verificada con `eval.py` en `loop_empty`, `episodes=1`,
> `device=cpu`, `init-order=model-first` → sin error, `length 1500.0`.

## Resultados finales de Fase 3

Tabla comparativa completa (Colab, GPU; `eval.py` con `--init-order model-first`, 3
episodios por mapa). Formato: `recompensa media ± std / longitud media`.
`loop_obstacles` es el mapa oculto, usado **solo para evaluación final** (con
`--allow-eval-hidden`), **nunca para entrenamiento**.

| modelo / stage | loop_empty | small_loop | loop_obstacles | decisión |
|---|---|---|---|---|
| DQN 20k | −1013.352 ± 27.031 / 112.3 | −1001.872 ± 46.784 / 86.3 | −1065.766 ± 20.149 / 61.0 | descartado |
| DQN 50k | −1047.611 ± 76.916 / 82.0 | −1016.568 ± 65.748 / 74.3 | −1032.042 ± 52.425 / 36.7 | descartado |
| **PPO 20k** | **960.964 ± 605.380 / 1500** | **317.489 ± 703.306 / 1500** | **1118.216 ± 488.214 / 1500** | **GANADOR (best_agent.zip)** |
| PPO 50k | −813.731 ± 210.545 / 1500 | −1238.875 ± 393.613 / 1500 | −1105.378 ± 760.645 / 1500 | descartado (degrada) |
| SAC 20k | 990.616 ± 139.264 / 1500 | −1253.567 ± 120.299 / 295.7 | 915.236 ± 571.946 / 1500 | no supera a PPO 20k |
| SAC 50k | — | — | — | no completado (interrumpido ~30k ts) |
| PPO avanzado v1 (ppo_adv20k) | −997.117 ± 604.477 | −1194.933 ± 1355.062 | −1538.882 ± 911.956 | descartado |
| PPO avanzado v1 (ppo_adv50k) | — | — | −1078.173 ± 162.939 | descartado |
| PPO avanzado v2 (ppo_adv_v2_20k) | −188.287 ± 1720.889 | −84.740 ± 1053.777 | 138.729 ± 1614.885 | no supera a PPO 20k |
| PPO fine-tuned (ppo_finetuned_5k) | −364.158 ± 1791.466 | 821.278 ± 328.210 | 513.225 ± 1172.583 | no supera a PPO 20k |

### Decisión final (honesta)
La **Fase 3 se implementó y evaluó con varias variantes avanzadas** —PPO avanzado v1
(hiperparámetros agresivos), PPO avanzado v2 (conservador, `ent_coef=0.001`) y PPO
**fine-tuning** desde el ganador (`--init-model`, `lr=5e-5`)—, además de SAC. **Ninguna
superó al PPO 20k baseline en el criterio principal**: la recompensa media en el mapa
oculto `loop_obstacles`.

- **PPO 20k** es el mejor en `loop_obstacles` (**1118.216**) y positivo en los tres mapas.
- **PPO 50k** degrada (recompensas negativas): más entrenamiento no ayuda.
- **SAC 20k** es competitivo en `loop_empty`/`loop_obstacles` (915.236) pero generaliza mal
  a `small_loop`; **SAC 50k no se completó** (interrumpido en Colab ~30k ts, sin artefacto).
- **PPO avanzado v1** se descarta (recompensas muy negativas). **v2** mejora respecto a v1
  pero queda lejos del baseline. El **fine-tuning** mejora `small_loop` (821.278) pero su
  `loop_obstacles` (513.225) **no alcanza** al PPO 20k (1118.216).

**Por tanto, el modelo final se mantiene como `best_agent.zip` = PPO 20k**
(`ppo_loop_empty_20k_gpu`). No se sustituye.
