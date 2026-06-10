# EXPERIMENTS_MOVEMENT.md — Experimento de movimiento (rama `movement-reward-fix`)

> **Documento experimental, NO forma parte de la entrega final.** Vive solo en la rama
> `movement-reward-fix` y **no se mergea a `main`**. El modelo final de la entrega sigue
> siendo **PPO 20k** (`best_agent.zip`), sin cambios. Ver `EXPERIMENTS.md` (en `main`)
> para los experimentos oficiales de la Fase 3.

## 1. Motivación: poco movimiento visible en el vídeo

Al generar el vídeo de evaluación del modelo final (PPO 20k) en `loop_empty` con seed 42
se observó **muy poco desplazamiento visible**: el agente no estaba completamente parado,
pero recorría una distancia muy baja para un episodio largo.

Medición del rollout (loop_empty, seed 42):

| Métrica | Valor |
|---|---|
| reward (limpio) | ≈ 503.67 |
| frames | 1501 |
| distancia recorrida | ≈ 0.5144 |

La distancia ≈ 0.51 aparecía **solo al final**, por el `reset` del entorno al cambiar de
episodio (la posición saltaba de `pos≈[3.7278, 0, 2.4299]` a `pos≈[3.3085, 0, 2.7278]`),
no por conducción real.

## 2. Diagnóstico: acción continua degenerada

Midiendo las acciones del modelo dentro de `make_eval_video.py`:

```
mean_action = [-0.54141094, 0.07327359]
std_action  = [ 0.00244333, 0.00047426]
```

La **primera componente (velocidad/avance) era ≈ -0.54**, casi constante. Con el wrapper
original (`action_space` de velocidad en `[-1, 1]`), `_normalize_action` recortaba esa
velocidad negativa a `0.0` → el robot **no avanzaba**, aunque la recompensa limpia pudiera
acumular valores positivos. Es decir, PPO había aprendido una **política degenerada**:
mandar velocidad negativa (= parado tras el clip).

## 3. Intervenciones probadas

Todas en la rama `movement-reward-fix` (sin tocar `main` ni `best_agent.zip`):

1. **Reward shaping de movimiento** (`DuckieWrapper`): bonus por desplazamiento real
   (`movement_bonus * step_dist`), penalización por quedarse parado y terminación del
   episodio tras demasiados pasos quietos. Flag `--disable-movement-shaping` para evaluar
   con la recompensa **limpia** del simulador (comparación justa).
2. **`min_forward_speed`** (defensa externa): fuerza `velocidad ≥ 0.1` justo antes de
   pasar la acción al simulador. Stage `ppo_move20k` (`min_forward_speed=0.1`).
3. **`continuous_min_speed`** (espacio de acción): el `action_space` de velocidad arranca
   en `0.1` (`Box([0.1, -1], [1, 1])`), de modo que PPO ve que la velocidad mínima ya es
   positiva. Stage `ppo_forward5k`/`ppo_forward20k` (`continuous_min_speed=0.1`).

También se acotó la velocidad del `action_space` a `[0, 1]` (no negativa) como base de
todos los experimentos de esta rama.

## 4. Resultados: se mueven más, pero no conducen

Los modelos se desplazaban más que el PPO 20k, pero **se salían de la carretera** y
obtenían **recompensas muy negativas** (evaluación limpia):

| Modelo | loop_empty | small_loop | Observaciones |
|---|---|---|---|
| `ppo_movement_20k` | reward ≈ -1186.0 | — | distancia ≈ 2.30, frames ≈ 611, `mean_action ≈ [0.0, -0.0933]` (avance solo por la restricción externa, no aprendido) |
| `ppo_forward5k` | reward ≈ -1105 | reward ≈ -1026 | deriva clara; muchos eventos *not drivable* / *Invalid pose* |

Conclusión técnica: **forzar movimiento ≠ aprender a conducir**. La restricción de avance
(externa o en el espacio de acción) garantiza desplazamiento, pero la política no aprende
una conducción estable; el agente avanza y se sale de la pista.

## 5. Decisión final

- **Descartar** los modelos `ppo_movement_20k` y `ppo_forward5k` (y no entrenar
  `ppo_forward20k`).
- **Conservar PPO 20k original como `best_agent.zip`** (entrega final estable, intacta).
- **No mergear** `movement-reward-fix` a `main` salvo aprobación explícita. Esta rama
  queda como **registro honesto** del diagnóstico y de los intentos de mejora.

El código de esta rama (reward shaping, `--min-forward-speed`, `--continuous-min-speed`,
stages `ppo_move*`/`ppo_forward*`) se mantiene reproducible por si se quisiera retomar la
línea con un enfoque distinto (p. ej. penalizar salirse de la pista en lugar de —o además
de— forzar avance).
