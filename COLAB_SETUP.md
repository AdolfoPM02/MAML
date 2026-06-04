# COLAB_SETUP.md — Receta reproducible para Duckietown + SB3 en Google Colab

Guía paso a paso para entrenar y evaluar los agentes (DQN/PPO/SAC) en **Duckietown
real** sobre **Google Colab**, respetando el contrato de evaluación del reto.

> El entrenamiento real **no** puede hacerse en local por el conflicto de dependencias
> de Duckietown (`numpy<=1.20`). Colab es el entorno objetivo. El profesor pide
> **Python 3.11**; como el kernel de Colab es 3.12, montamos un **venv 3.11** y
> ejecutamos todo como subprocess con `PY`.
>
> El notebook ejecutable equivalente está en `notebooks/colab_smoke.ipynb`.

**Convenciones usadas en todos los comandos:**
```bash
PY=/content/venv-maml/bin/python   # intérprete del venv Python 3.11
```
- El repo se clona **siempre** en `/content/MAML`. Antes de ejecutar cualquier script
  hay que estar ahí: `%cd /content/MAML` (ver error 2).
- Todos los comandos que usan `{PY}` se anteponen con `MPLBACKEND=Agg` (ver error 3).

---

## ⚠️ Errores reales encontrados en Colab y sus correcciones

Esta receta incorpora las correcciones de cinco errores observados en una ejecución
real del notebook:

1. **Clonado del repo privado (403 Write access not granted).** El PAT no tenía
   permiso. → El PAT debe tener scope **`repo`** si el repositorio es privado;
   alternativa recomendada: **hacer el repo público temporalmente** y clonar sin PAT.
   La celda de clonado ahora borra `/content/MAML`, clona explícitamente ahí, hace
   `%cd` y lista `scripts/` y `src/` para confirmar.
2. **Ejecutar desde `/content` en vez de `/content/MAML`** (`can't open file
   '/content/train.py'`, etc.). → `%cd /content/MAML` antes de cada bloque que ejecuta
   scripts (o rutas absolutas).
3. **Matplotlib backend al importar SB3** (`Key backend:
   'module://matplotlib_inline.backend_inline' is not a valid value`). El subprocess
   heredaba `MPLBACKEND` del kernel. → Fijar `MPLBACKEND=Agg` (celda + prefijo en cada
   comando con `{PY}`).
4. **`ModuleNotFoundError: zuper_commons`** tras instalar Duckietown con `--no-deps`.
   → Instalar las dependencias **reales** de daffy (NO existe `zuper-ipce` en PyPI):
   `zuper-commons-z6`, `duckietown-world-daffy`, `PyGeometry-z6`, `carnivalmirror`, etc.
5. **`ModuleNotFoundError: gym`** (gym_duckietown usa el `gym` antiguo). → Instalar
   `gym==0.26.2` junto a `gymnasium`, y re-fijar `numpy==1.26.4` después.

---

## 1. Clonar el repositorio en `/content/MAML`

> **Siempre** `%cd /content` antes de `rm -rf /content/MAML`: si una celda previa dejó
> el cwd dentro de `/content/MAML`, borrar ese directorio dejaría el shell en una ruta
> inexistente (ver troubleshooting: `getcwd: cannot access parent directories`).

**Si el repo es PRIVADO** (el PAT debe tener scope **`repo`**):
```python
from getpass import getpass
import os

token = getpass("GitHub PAT con permiso repo: ")
repo_url = f"https://AdolfoPM02:{token}@github.com/AdolfoPM02/MAML.git"

%cd /content
!rm -rf /content/MAML
!git clone {repo_url} /content/MAML
%cd /content/MAML
!pwd
!ls -la
!ls scripts
!ls src
```

**Alternativa recomendada — repo PÚBLICO temporal** (sin PAT, más simple y sin riesgo
de filtrar el token):
```python
%cd /content
!rm -rf /content/MAML
!git clone https://github.com/AdolfoPM02/MAML.git /content/MAML
%cd /content/MAML
!pwd
!ls -la
!ls scripts
!ls src
```
`ls scripts` y `ls src` deben mostrar los archivos; si no, el clonado falló y el resto
de celdas no encontrará los scripts.

---

## 2. Python 3.11 + venv + dependencias de sistema

```bash
!sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt-get update -qq
!sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev \
     xvfb python3-opengl ffmpeg freeglut3-dev
!python3.11 -m venv /content/venv-maml
!/content/venv-maml/bin/python -m pip install -U pip wheel setuptools
```
En el notebook se define `PY = "/content/venv-maml/bin/python"`.

---

## 3. Instalar stack ML + Duckietown real (sin romper numpy)

El **orden importa**. numpy `1.26.4` es el "puente" que satisface a SB3/torch/gymnasium
y, en la práctica, también a Duckietown (pese a su pin declarado `<=1.20`).

```bash
PY=/content/venv-maml/bin/python
# a) Stack moderno (mismas versiones con las que se desarrolló el código)
!{PY} -m pip install "numpy==1.26.4" "stable-baselines3==2.8.0" torch \
     "opencv-python" "gymnasium==1.2.3" pyvirtualdisplay

# b) Duckietown SIN deps (para que no arrastre numpy<=1.20)
!{PY} -m pip install --no-deps "git+https://github.com/duckietown/gym-duckietown.git@daffy"

# c) Dependencias REALES de gym-duckietown daffy (NO usar zuper-ipce: no existe en PyPI)
!{PY} -m pip install \
  "zuper-commons-z6" \
  "duckietown-world-daffy" \
  "PyGeometry-z6" \
  "carnivalmirror==0.6.2" \
  "pyzmq>=16.0.0" \
  "PyYAML>=3.11" \
  "Pillow" \
  "typing_extensions" \
  "pyglet==1.5.27"

# d) gym ANTIGUO: gym_duckietown hace `from gym.envs.registration import register`
!{PY} -m pip install "gym==0.26.2"

# e) Re-fijar numpy por si algún paso lo cambió
!{PY} -m pip install "numpy==1.26.4" --force-reinstall --no-deps
```

> `pyglet==1.5.27` se mantiene porque es necesario para el render OpenGL de Duckietown.
> Las versiones son **candidatas**: se afinan en el primer run real y luego se
> congelan en `requirements.txt` (paso 10).

---

## 4. Verificar imports (gym **y** gymnasium)

```bash
!MPLBACKEND=Agg {PY} -c "import numpy, torch, cv2, gym, gymnasium, stable_baselines3; \
import gym_duckietown; \
print('numpy', numpy.__version__, 'torch', torch.__version__); \
print('gym', gym.__version__, 'gymnasium', gymnasium.__version__); \
print('duckietown OK')"
```
Debe imprimir `numpy 1.26.4 ...`, las versiones de gym/gymnasium y `duckietown OK`.

---

## 5. Smoke tests con MOCK (no requieren display)

```bash
%cd /content/MAML
!MPLBACKEND=Agg {PY} scripts/smoke_test_phase2.py
!MPLBACKEND=Agg {PY} scripts/smoke_test_model_load.py
```
Esperado: 12/12 checks y "CONTRATO DE CARGA: TODO OK". Si esto falla, el problema es
del stack ML (no de Duckietown ni del display).

---

## 6. Duckietown REAL — `reset()`

```bash
%cd /content/MAML
!MPLBACKEND=Agg xvfb-run -a -s "-screen 0 1024x768x24" {PY} scripts/check_duckie_real.py --reset-only
```
Esperado: `reset() entorno base -> RGB (480, 640, 3)`. Aquí se valida que el
contexto OpenGL/headless funciona.

---

## 7. Wrappers + shapes con Duckietown REAL

```bash
%cd /content/MAML
!MPLBACKEND=Agg xvfb-run -a {PY} scripts/check_duckie_real.py
```
Esperado: `DuckieWrapper -> (1, 64, 64)` y `build_vec_env -> obs_space (4, 64, 64)`.

---

## 8A. Diagnóstico runtime de Duckietown (aislar segfaults)

El PPO real crashea con `Segmentation fault (core dumped)` **incluso forzando CPU**, lo
que indica un fallo NATIVO (Duckietown/OpenGL/pyglet/xvfb), no de CUDA.
`scripts/debug_duckie_runtime.py` ejecuta niveles crecientes de aislamiento para ver
**en qué fase** ocurre: step base → VecFrameStack → SB3 init → SB3 learn.

```bash
%cd /content/MAML
# A) entorno base + steps reales
!env MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" xvfb-run -a {PY} scripts/debug_duckie_runtime.py --steps 20 --device cpu
# B) build_vec_env + VecFrameStack + steps
!env MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" xvfb-run -a {PY} scripts/debug_duckie_runtime.py --vec --steps 20 --device cpu
# C) construir PPO con CustomCNN (sin learn)
!env MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" xvfb-run -a {PY} scripts/debug_duckie_runtime.py --sb3-init --device cpu
# D) PPO.learn muy corto
!env MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" xvfb-run -a {PY} scripts/debug_duckie_runtime.py --sb3-learn --timesteps 64 --device cpu
```
Cada comando imprime la fase alcanzada (`reset`, `step loop`, `vec reset`, `sb3 init`,
`sb3 learn`). La última fase impresa antes del segfault localiza el componente culpable.
**Solo pasa a la sección 8 si el modo D (`--sb3-learn`) termina con `FIN OK`.**

---

## 8B. Diagnóstico SB3 init (el crash ocurre al construir PPO)

El diagnóstico 8A mostró que el segfault ocurre en la fase **`sb3 init`** (construir PPO
con el entorno real), no en step/VecFrameStack/learn. Estos comandos separan **SB3/torch/
CNN** del **entorno real**, usando un entorno SINTÉTICO con espacios idénticos:

```bash
%cd /content/MAML
# A) synthetic + CnnPolicy + CustomCNN  (¿SB3 + nuestra CNN + spaces están bien?)
!env MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" xvfb-run -a {PY} scripts/debug_duckie_runtime.py --synthetic-env --sb3-init --device cpu
# B) synthetic + CnnPolicy SIN CustomCNN  (¿influye CustomCNN?)
!env MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" xvfb-run -a {PY} scripts/debug_duckie_runtime.py --synthetic-env --sb3-init --no-custom-cnn --device cpu
# D) Duckietown real: solo imprimir spaces/reset (sin PPO)
!env MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" xvfb-run -a {PY} scripts/debug_duckie_runtime.py --check-spaces --device cpu
# C) Duckietown real + CnnPolicy SIN CustomCNN  (¿depende de CustomCNN o del entorno real?)
!env MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" xvfb-run -a {PY} scripts/debug_duckie_runtime.py --sb3-init --no-custom-cnn --device cpu
```

**Interpretación:**
- **A pasa** → SB3 + CustomCNN + spaces están bien (el problema NO es nuestra CNN ni SB3).
- **A falla** → el problema está en SB3/torch/CNN/versiones (no en Duckietown).
- **C falla** (real sin CustomCNN) → el fallo está en el **entorno real + SB3 init**, no en
  CustomCNN. Apunta a la interacción entre el contexto OpenGL/pyglet de Duckietown y la
  inicialización de torch (p. ej. orden de import, hilos, contexto GL global).
- **check-spaces raro** (dtype/shape/low/high inesperados) → corregir espacios/wrappers.

> Pista típica para C-falla: el contexto OpenGL de Duckietown y la inicialización de
> torch pueden chocar. Posibles vías a probar en una iteración posterior: construir el
> entorno **después** de importar torch, o crear PPO **antes** de tocar el render, o
> aislar el render con `PYOPENGL_PLATFORM`.

---

## 8C. Últimos fixes: backend OpenGL y orden de inicialización

El diagnóstico 8B confirmó que el segfault ocurre al **inicializar PPO con el entorno
Duckietown real** (no CustomCNN, no spaces, no learn). Dos vías de fix controladas:

**(1) Cambiar el backend OpenGL** (variable de entorno, no se hardcodea en Python):

```bash
%cd /content/MAML
# A) EGL + real sin CustomCNN
!env MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" PYOPENGL_PLATFORM=egl xvfb-run -a {PY} scripts/debug_duckie_runtime.py --sb3-init --no-custom-cnn --device cpu
# B) OSMesa + real sin CustomCNN (instala libosmesa6 si falta: !sudo apt-get install -y libosmesa6)
!env MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" PYOPENGL_PLATFORM=osmesa xvfb-run -a {PY} scripts/debug_duckie_runtime.py --sb3-init --no-custom-cnn --device cpu
```

**(2) Cambiar el orden de inicialización** (`--init-order model-first`: construir PPO
sobre un env sintético compatible y luego `set_env(real)`):

```bash
# C) model-first sin CustomCNN
!env MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" xvfb-run -a {PY} scripts/debug_duckie_runtime.py --sb3-init --no-custom-cnn --device cpu --init-order model-first
# D) model-first con CustomCNN
!env MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" xvfb-run -a {PY} scripts/debug_duckie_runtime.py --sb3-init --device cpu --init-order model-first
```

**Interpretación / decisión:**
- Si **EGL** o **OSMesa** dan `FIN OK` → usar ese `PYOPENGL_PLATFORM` también al entrenar
  (anteponerlo en los comandos de la sección 8).
- Si **model-first** da `FIN OK` (`set_env OK`) → la solución es inicializar PPO antes de
  tocar Duckietown; en una fase posterior se adaptará `train.py` para soportar ese flujo
  (NO se toca `train.py` todavía).
- Si **todo** sigue en segfault → el fallo está en la integración nativa SB3+Duckietown
  en Colab (xvfb/pyglet/OpenGL). **Plan B**: documentar y considerar alternativas (otro
  runtime/imagen, `gym-duckietown` en proceso separado del de entrenamiento, o reducir el
  alcance a evaluar un modelo provisto), a decidir contigo.

---

## 8. Entrenar un PPO corto real (CPU forzada)

> **Ejecutar solo si los diagnósticos 8A/8B/8C no crashean** (con el `PYOPENGL_PLATFORM`
> y/o `--init-order` que haya resultado estable en 8C). Si todo crashea, ver el Plan B
> de 8C antes de continuar.
> Forzamos **CPU**: Duckietown/OpenGL/xvfb + PyTorch CUDA provoca segfault (ver
> troubleshooting). La prioridad del contrato es que **cargue y ejecute**.

```bash
%cd /content/MAML
!env MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" xvfb-run -a {PY} train.py \
  --algo ppo \
  --map Duckietown-loop_empty-v0 \
  --timesteps 512 \
  --output ppo_colab_test \
  --device cpu
```
Comprobar que el modelo se guardó:
```bash
!ls -lh models/
```
Debe aparecer `ppo_colab_test.zip`. Si no está, la sección 8 crasheó antes de guardar.

---

## 9. Evaluar el modelo real (CPU forzada)

> **Ejecutar solo si existe `models/ppo_colab_test.zip`** (confirmado en el paso 8 con
> `ls -lh models/`). Si la sección 8 crasheó, este paso dará
> `FileNotFoundError: models/ppo_colab_test.zip`.

```bash
%cd /content/MAML
!env MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" xvfb-run -a {PY} eval.py \
  --algo ppo \
  --model models/ppo_colab_test \
  --map Duckietown-loop_empty-v0 \
  --episodes 1 \
  --device cpu
# Prueba del contrato en el mapa oculto (solo evaluación):
!env MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" xvfb-run -a {PY} eval.py \
  --algo ppo \
  --model models/ppo_colab_test \
  --map Duckietown-loop_obstacles-v0 \
  --episodes 1 --allow-eval \
  --device cpu
```
Imprime recompensa acumulada media ± std y longitud media. El entrenamiento en
`loop_obstacles` sigue bloqueado (solo `--allow-eval` lo habilita, y solo para evaluar).

---

## 10. Preparar la entrega (cuando el entrenamiento completo esté listo)

> **Aún no se hace**: requiere el entrenamiento real completo. Documentado para después.

```bash
%cd /content/MAML
# Copiar el mejor modelo al nombre EXACTO del contrato
!cp models/<mejor_modelo>.zip models/best_duckie_agent.zip
# Congelar dependencias con versiones exactas (==)
!{PY} -m pip freeze > requirements.txt
```
Revisar que `requirements.txt` incluya: `stable-baselines3==2.8.0`, `torch==...`,
`gymnasium==1.2.3`, `gym==0.26.2`, `numpy==1.26.4`, `gym-duckietown` (línea de git),
`opencv-python`, `pyglet==1.5.27`, `zuper-commons-z6`, `duckietown-world-daffy`,
`PyGeometry-z6`, `carnivalmirror==0.6.2`, `pyvirtualdisplay`. Hacer el **dry-run** del
contrato: Colab nuevo → instalar solo desde `requirements.txt` → cargar
`best_duckie_agent.zip` → evaluar.

---

## Troubleshooting

**403 al clonar.** PAT sin scope `repo` (privado) → regenerar PAT con `repo`, o hacer el
repo público temporalmente. Confirmar con `!ls scripts` tras clonar.

**`can't open file '/content/...py'`.** No estás en `/content/MAML`. Ejecutar
`%cd /content/MAML` (está al inicio de cada celda de ejecución).

**`getcwd: cannot access parent directories` / `Unable to read current working
directory` / `[Errno 2] No such file or directory: '/content/MAML'`.** Reejecutaste la
celda de clonado estando **dentro** de `/content/MAML` y `rm -rf /content/MAML` borró el
directorio actual del shell. → Hacer **siempre** `%cd /content` antes de
`rm -rf /content/MAML` (ya incluido en las celdas de clonado).

**`Key backend: '...backend_inline' is not a valid value`.** Falta `MPLBACKEND=Agg`.
Está como celda (`os.environ["MPLBACKEND"]="Agg"`) y como prefijo de cada comando `{PY}`.

**`ModuleNotFoundError: zuper_commons`.** Faltan deps de Duckietown por el `--no-deps`.
Instalar el bloque del paso 3c. **No** instalar `zuper-ipce` (no existe en PyPI).

**`ModuleNotFoundError: gym`.** Falta el gym antiguo. `!{PY} -m pip install "gym==0.26.2"`
y re-fijar `numpy==1.26.4`.

**numpy se rompe / `gym_duckietown` no importa.** Reejecutar el paso 3e
(`numpy==1.26.4 --force-reinstall --no-deps`) y `!{PY} -c "import numpy; print(numpy.__version__)"`.
Verificar numpy tras cada bloque de instalación.

**`Using cuda device` → `Segmentation fault (core dumped)`** al entrenar/evaluar
Duckietown real. La combinación Duckietown/OpenGL/xvfb + PyTorch CUDA crashea en Colab.
→ Forzar **CPU** en los smoke tests reales: `--device cpu` y anteponer
`CUDA_VISIBLE_DEVICES=""` (`!env MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" xvfb-run -a {PY}
...`). La GPU se podrá estudiar más adelante; la prioridad del contrato es que el
modelo **cargue y ejecute**.

**`Using cpu device` → `Segmentation fault (core dumped)`** (el segfault persiste en
CPU). **No es CUDA**: es un fallo nativo, probablemente de Duckietown/OpenGL/pyglet/xvfb
durante `step`/`learn`. → Usar `scripts/debug_duckie_runtime.py` (sección 8A) para
aislar si falla en el step real (modo A), en VecFrameStack (B), en la construcción de
SB3 (C) o en `learn` (D). La última fase impresa antes del segfault indica el componente.
No ejecutar la sección 8 (PPO real) hasta que el modo D termine con `FIN OK`.

**`ValueError: setting an array element with a sequence`** dentro de `env.step` de
Duckietown (en `pwm_dynamics`/`se2_from_linear_angular`). **Causa**: SB3/DummyVecEnv
entrega la acción con una dimensión extra (p. ej. `(1, 2)`); Duckietown espera un vector
plano de 2 escalares `[velocidad, giro]`. **Solución**: ya resuelto —
`DuckieWrapper.step` normaliza la acción a `(2,)` (aplana, valida 2 valores y recorta
velocidad a `[0,1]` y giro a `[-1,1]`) antes de llamar a `self.env.step`.

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
