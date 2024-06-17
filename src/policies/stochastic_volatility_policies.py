#!/usr/bin/env python3

from typing import List
import torch
import torch.distributions as dist

from src.policies.base_policy import Policy


class SVPrior(Policy):
    def __init__(
        self,
        dim: int,
        alpha: float,
        sigma: float,
        **kwargs
    ):
        super().__init__(dim)
        self.alpha = alpha
        self.sigma = sigma

    def x0_dist(self) -> torch.Tensor:
        std = self.sigma / torch.tensor((1 - self.alpha ** 2)).sqrt()
        return dist.Independent(
            dist.Normal(
                torch.zeros(self.dim),
                std * torch.ones(self.dim)
            ),
            reinterpreted_batch_ndims=1
        )

    def xt_dist(self, x_prev: torch.Tensor,) -> torch.Tensor:
        mean = self.alpha * x_prev
        return dist.Independent(
            dist.Normal(mean, self.sigma * torch.ones(self.dim)),
            reinterpreted_batch_ndims=1
        )

    def sample_x0(self, num_particles: int) -> torch.Tensor:
        return self.x0_dist().sample([num_particles])

    def sample_xt(self, x_prev: torch.Tensor) -> torch.Tensor:
        return self.xt_dist(x_prev).sample()

    def sample(
            self,
            state: torch.Tensor,
            t: int,
            num_particles: int
    ) -> torch.Tensor:
        if t == 0:
            samples = self.sample_x0(num_particles)
        else:
            # we don't need to pass num_particles
            # since the state already has shape
            # (num_particles x dim)
            samples = self.sample_xt(state)
        return samples, self.log_prob(samples, t, state)

    def log_prob(
            self,
            action: torch.Tensor,
            t: int,
            state: torch.Tensor
    ) -> torch.Tensor:
        if t == 0:
            return self.x0_dist().log_prob(action)
        return self.xt_dist(state).log_prob(action)
