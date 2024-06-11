#!/usr/bin/env python3

import torch
import torch.distributions as dist


class Policy:
    def __init__(self, dim: int, **kwargs):
        self.dim = dim
        self.dist = dist.MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))

    def sample(self, obs: torch.Tensor, num_particles: int):
        return self.dist.rsample([num_particles])
