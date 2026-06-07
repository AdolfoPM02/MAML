# SAC_RESULTS.md — Validación experimental de la Fase 3 (SAC)

Validación de **SAC** (Soft Actor-Critic) en la rama `sac-validation`, sin tocar la
entrega estable de `main` (modelo ganador actual: PPO 20k). Objetivo: tener resultados
reales de SAC (5k y, si va bien, 20k) para justificar honestamente la Fase 3.

> Reglas: `--init-order model-first`; `Duckietown-loop_obstacles-v0` **solo evaluación**
> (`--allow-eval-hidden`), nunca entrenamiento. El lanzador hace **SKIP** del
> entrenamiento si el `.zip` ya existe (usar `--overwrite` para rehacer).

## Comandos (Colab, GPU)
```bash
%cd /content/MAML
# 1) SAC 5k (prueba de estabilidad)
!{PY} scripts/run_training_plan.py --stage sac5k --execute --eval-after --use-gpu --device cuda
# 2) Si 5k termina bien, SAC 20k
!{PY} scripts/run_training_plan.py --stage sac20k --execute --eval-after --use-gpu --device cuda
# 3) Evaluar SAC 20k también en el mapa oculto (SKIP de entrenamiento; solo evalúa)
!{PY} scripts/run_training_plan.py --stage sac20k --execute --eval-after --allow-eval-hidden --use-gpu --device cuda
```

## Resultados (rellenar)

| stage  | device | timesteps | tiempo | loop_empty (rew/len) | small_loop (rew/len) | loop_obstacles (rew/len) | observaciones |
|--------|--------|-----------|--------|----------------------|----------------------|--------------------------|---------------|
| sac5k  | GPU    |  5 000    |        |                      |                      | —                        |               |
| sac20k | GPU    | 20 000    |        |                      |                      |                          |               |

## Conclusión (rellenar tras los experimentos)
- ¿SAC entrena de forma estable en Duckietown?
- ¿Mejora, iguala o queda por debajo del PPO 20k ganador (loop_obstacles 1118.216)?
- Decisión: mantener PPO 20k como `best_agent.zip` o reconsiderar.
