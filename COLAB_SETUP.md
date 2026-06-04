# COLAB_SETUP.md — Receta reproducible para Duckietown + SB3 en Google Colab

Guía paso a paso para entrenar y evaluar los agentes (DQN/PPO/SAC) en **Duckietown
real** sobre **Google Colab**, respetando el contrato de evaluación del reto.

> El entrenamiento real **no** puede hacerse en local por el conflicto de dependencias
> de Duckietown (`numpy<=1.20`). Colab es el entorno objetivo. El profesor pide
> **Python 3.11**; como el kernel de Colab es 3.12, montamos un **venv 3.11** y
> ejecutamos todo como subprocess con ese intérprete.
>
> El notebook ejecutable equivalente está en `notebooks/colab_smoke.ipynb`.

**Convención usada en todos los comandos:**
```bash
PY=/content/venv-maml/bin/python   # intérprete del venv Python 3.11
```

---

## 1. Clonar el repositorio (privado)

El repo `AdolfoPM02/MAML` es privado. Tres opciones:

**a) Token en la URL (rápido):**
```python
from getpass import getpass
import os
os.environ["GH_PAT"] = getpass("GitHub PAT: ")
!git clone https://$GH_PAT@github.com/AdolfoPM02/MAML.git
%cd MAML
```
**b) `gh` CLI** autenticado con el PAT. **c)** Hacer el repo público temporalmente y
`git clone https://github.com/AdolfoPM02/MAML.git`.

---

## 2. Python 3.11 + venv + dependencias de sistema

```bash
!sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt-get update -qq
!sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev \
     xvfb python3-opengl ffmpeg freeglut3-dev
!python3.11 -m venv /content/venv-maml
!/content/venv-maml/bin/python -m pip install -U pip wheel setuptools
```

---

## 3. Instalar stack ML + Duckietown real (sin romper numpy)

El **orden importa**. numpy `1.26.4` es el "puente" que satisface a SB3/torch/gymnasium
y, en la práctica, también a Duckietown (pese a su pin declarado `<=1.20`).

```bash
PY=/content/venv-maml/bin/python
# a) Stack moderno (mismas versiones con las que se desarrolló el código)
!$PY -m pip install "numpy==1.26.4" "stable-baselines3==2.8.0" torch \
     "opencv-python" "gymnasium==1.2.3" pyvirtualdisplay
# b) Duckietown SIN deps (para que no arrastre numpy<=1.20)
!$PY -m pip install --no-deps "git+https://github.com/duckietown/gym-duckietown.git@daffy"
# c) Dependencias de runtime de Duckietown (salvo numpy)
!$PY -m pip install "pyglet==1.5.27" pyzmq PyYAML scikit-image pillow
# d) Re-fijar numpy por si algún paquete lo cambió
!$PY -m pip install "numpy==1.26.4" --force-reinstall --no-deps
```

> Las versiones son **candidatas**: se afinan en el primer run real y luego se
> congelan en `requirements.txt` (paso 10).

---

## 4. Verificar imports

```bash
!$PY -c "import numpy,torch,cv2,gymnasium,stable_baselines3; \
import gym_duckietown; print('numpy',numpy.__version__,'torch',torch.__version__); \
print('duckietown OK')"
```
Debe imprimir `numpy 1.26.4 ...` y `duckietown OK`.

---

## 5. Smoke tests con MOCK (no requieren display)

```bash
!$PY scripts/smoke_test_phase2.py
!$PY scripts/smoke_test_model_load.py
```
Esperado: 12/12 checks y "CONTRATO DE CARGA: TODO OK". Si esto falla, el problema es
del stack ML (no de Duckietown ni del display).

---

## 6. Duckietown REAL — `reset()`

```bash
!xvfb-run -a -s "-screen 0 1024x768x24" $PY scripts/check_duckie_real.py --reset-only
```
Esperado: `reset() entorno base -> RGB (480, 640, 3)`. Aquí se valida que el
contexto OpenGL/headless funciona.

---

## 7. Wrappers + shapes con Duckietown REAL

```bash
!xvfb-run -a $PY scripts/check_duckie_real.py
```
Esperado: `DuckieWrapper -> (1, 64, 64)` y `build_vec_env -> obs_space (4, 64, 64)`.

---

## 8. Entrenar un PPO corto real

```bash
!xvfb-run -a $PY train.py --algo ppo --map Duckietown-loop_empty-v0 \
     --timesteps 5000 --output ppo_colab_test
```
Genera `models/ppo_colab_test.zip`. (Con GPU, `--device auto` la usará.)

---

## 9. Evaluar el modelo real

```bash
!xvfb-run -a $PY eval.py --algo ppo --model models/ppo_colab_test \
     --map Duckietown-loop_empty-v0 --episodes 3
# Prueba del contrato en el mapa oculto (solo evaluación):
!xvfb-run -a $PY eval.py --algo ppo --model models/ppo_colab_test \
     --map Duckietown-loop_obstacles-v0 --episodes 3 --allow-eval
```
Imprime recompensa acumulada media ± std y longitud media. El entrenamiento en
`loop_obstacles` sigue bloqueado (solo `--allow-eval` lo habilita, y solo para evaluar).

---

## 10. Preparar la entrega (cuando el entrenamiento completo esté listo)

> **Aún no se hace**: requiere el entrenamiento real completo. Documentado para después.

```bash
# Copiar el mejor modelo al nombre EXACTO del contrato
!cp models/<mejor_modelo>.zip models/best_duckie_agent.zip
# Congelar dependencias con versiones exactas (==)
!$PY -m pip freeze > requirements.txt
```
Revisar que `requirements.txt` incluya: `stable-baselines3==2.8.0`, `torch==...`,
`gymnasium==1.2.3`, `numpy==1.26.4`, `gym-duckietown` (línea de git), `opencv-python`,
`pyglet==1.5.27`, `pyvirtualdisplay`. Hacer el **dry-run** del contrato: Colab nuevo →
instalar solo desde `requirements.txt` → cargar `best_duckie_agent.zip` → evaluar.

---

## Troubleshooting

**numpy se rompe / `gym_duckietown` no importa.** Reejecutar el paso 3d
(`numpy==1.26.4 --force-reinstall --no-deps`) y `!$PY -c "import numpy; print(numpy.__version__)"`.
Verificar numpy tras cada bloque de instalación.

**OpenGL / display.**
- Confirmar `xvfb python3-opengl freeglut3-dev` instalados; lanzar con `xvfb-run -a`.
- Alternativa: `pyvirtualdisplay.Display(visible=False, size=(1024,768)).start()`.
- Si pyglet no crea contexto: `os.environ["PYOPENGL_PLATFORM"]="egl"`; si falla,
  `="osmesa"` (y `sudo apt-get install -y libosmesa6`).
- Último recurso: reiniciar el runtime tras instalar las deps de sistema.

**Orden de pruebas (de menor a mayor riesgo):** imports (4) → smoke mock (5) →
reset real (6) → wrappers/shapes (7) → PPO corto (8) → eval (9). Aísla si el fallo es
de dependencias, de display o del entorno.

---

## Qué NO cubre esta fase
- No se entrena el modelo completo ni se crea `best_duckie_agent.zip` definitivo.
- No se congela el `requirements.txt` final (se hará tras el entrenamiento real).
- No se modifica `Challenge_RL.ipynb` ni los notebooks de referencia.
