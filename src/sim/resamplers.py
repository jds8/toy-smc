#!/usr/bin/env python3

from abc import ABC, abstractmethod
import torch


class Resampler(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, weights: torch.Tensor, K: int):
        pass


class MultinomialResampler(Resampler):
    def __call__(self, weights: torch.Tensor, K: int):
        return torch.multinomial(weights, K, replacement=True)


class ResidualResampler(Resampler):
    def __call__(self, weights: torch.Tensor, K: int):
        """ lower variance than multinomial resampling; see Douc et al. 2005"""
        n = weights.shape[0]
        weights = weights / weights.sum()
        floor = torch.floor(n * weights)
        R = floor.sum()
        w_bar = (n * weights - floor) / (n - R)
        denom = (n - R).int().item()
        N_bar = torch.distributions.Multinomial(total_count=denom, probs=w_bar).sample()
        N = (floor + N_bar).int()
        idx = torch.arange(len(N)).repeat_interleave(N)
        return idx
