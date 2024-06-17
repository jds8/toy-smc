#!/usr/bin/env python3

from abc import ABC, abstractmethod
import torch


class Resampler(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, weights: torch.Tensor, K: int):
        pass


class NoResampler(Resampler):
    def __call__(self, weights: torch.Tensor, K: int):
        return torch.arange(weights.shape[0])


class MultinomialResampler(Resampler):
    def __call__(self, weights: torch.Tensor, K: int):
        return torch.multinomial(weights.squeeze(), K, replacement=True)


class ResidualResampler(Resampler):
    def __call__(self, weights: torch.Tensor, K: int):
        """ lower variance than multinomial resampling; see Douc et al. 2005"""
        assert len(weights.shape) == 1
        n = weights.shape[0]
        weights = weights / weights.sum()
        floor = torch.floor(n * weights)
        R = floor.sum()
        w_bar = (n * weights - floor) / (n - R)
        denom = (n - R).int().item()
        N_bar = torch.distributions.Multinomial(total_count=denom, probs=w_bar).sample()
        N = (floor + N_bar).int()
        idx = torch.arange(len(N)).repeat_interleave(N.squeeze())
        return idx


class StratifiedResampler(Resampler):
    def __call__(self, weights: torch.Tensor, K: int):
        """ lower variance than multinomial resampling; see Douc et al. 2005"""
        sets = torch.linspace(0, 1, K+1)
        us = torch.distributions.Uniform(sets[:-1], sets[1:]).rsample()
        normalized_weights = weights / weights.sum()
        idx = torch.searchsorted(normalized_weights.cumsum(dim=0), us)
        return idx


class SystematicResampler(Resampler):
    def __call__(self, weights: torch.Tensor, K: int):
        u = torch.distributions.Uniform(0., 1/K).rsample()
        us = torch.arange(0, K) / K + u
        normalized_weights = weights / weights.sum()
        idx = torch.searchsorted(normalized_weights.cumsum(dim=0), us)
        return idx
