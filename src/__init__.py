"""Paquete de Fase 2 — pipeline Duckietown + Stable-Baselines3.

Capa de abstracción para que el entorno MOCK (desarrollo local en Python 3.11) y
Duckietown real (Google Colab) compartan exactamente los mismos wrappers, CNN y
configuración. En esta fase NO se entrena: solo se valida la estructura y el
contrato de evaluación (observaciones (1,64,64) -> FrameStack -> (4,64,64)).
"""
