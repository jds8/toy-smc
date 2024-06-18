#!/usr/bin/env python3

from typing import Union, List

import torch
import torch.distributions as dist
from collections import namedtuple


class Policy:
    def __init__(self, dim: int, **kwargs):
        self.dim = torch.tensor(dim)

    def sample(self, obs: torch.Tensor, num_particles: int) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(
            self,
            action: torch.Tensor,
            t: int,
            obs: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError
