#!/usr/bin/env python3

import torch
from collections import namedtuple


class Policy:
    def __init__(self, dim: int, **kwargs):
        self.dim = torch.tensor(dim)

    def sample(self, obs: torch.Tensor, num_particles: int) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(
            self,
            action: torch.Tensor,
            obs: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError
