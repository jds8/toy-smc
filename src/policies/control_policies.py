#!/usr/bin/env python3

import stable_baselines3 as sb3
import torch
import torch.distributions as dist

from src.policies.linear_gaussian_policies import GaussianPolicy


class PriorPolicy(GaussianPolicy):
    def __init__(self, dim: int, **kwargs):
        A = torch.zeros(dim)
        Q = torch.eye(dim)
        super().__init__(A, Q)


class SigmoidPolicy(PriorPolicy):
    def __init__(self, dim: int, max_value: float, **kwargs):
        super().__init__(dim)
        self.base_dist = self.dist
        self.max_value = max_value
        affine_transform = dist.AffineTransform(
            loc=-self.max_value / torch.sqrt(self.dim),
            scale=2 * self.max_value / torch.sqrt(self.dim),
        )
        transforms = [dist.SigmoidTransform(), affine_transform]
        self.dist = dist.TransformedDistribution(self.base_dist, transforms)


class TanhPolicy(PriorPolicy):
    def __init__(self, dim: int, max_value: float, **kwargs):
        super().__init__(dim)
        self.base_dist = self.dist
        self.max_value = max_value
        affine_transform = dist.AffineTransform(
            loc=0,
            scale=self.max_value / torch.sqrt(self.dim)
        )
        transforms = [dist.TanhTransform(), affine_transform]
        self.dist = dist.TransformedDistribution(self.base_dist, transforms)
