"""CNN personalizada compatible con Stable-Baselines3 (Fase 2).

`CustomCNN` extrae características de la observación apilada (4, 64, 64). Se conecta
a las políticas de SB3 vía:

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

El número de canales de entrada se deriva del espacio de observación, por lo que
funciona tanto con (1, 64, 64) como con (4, 64, 64) tras VecFrameStack.
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from . import config


class CustomCNN(BaseFeaturesExtractor):
    """Extractor convolucional estilo NatureCNN para imágenes de Duckietown."""

    def __init__(self, observation_space: gym.spaces.Box,
                 features_dim: int = config.FEATURES_DIM):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # 4 tras FrameStack

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Dimensión aplanada calculada con un tensor dummy.
        with torch.no_grad():
            dummy = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(dummy).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Normalizar uint8 [0, 255] -> float [0, 1].
        x = observations.float() / 255.0
        return self.linear(self.cnn(x))
