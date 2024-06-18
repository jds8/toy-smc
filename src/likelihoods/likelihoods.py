#!/usr/bin/env python3

import torch
import torch.distributions as dist


class Likelihood:
    def __call__(self, state: torch.Tensor) -> dist.Distribution:
        raise NotImplementedError


class SVLikelihood(Likelihood):
    def __init__(self, beta):
        self.beta = beta

    def __call__(self, state: torch.Tensor) -> dist.Distribution:
        cov = self.beta * torch.exp(state / 2.)
        return dist.Independent(
            dist.Normal(torch.zeros_like(state), cov),
            reinterpreted_batch_ndims=1
        )


class LinearGaussianLikelihood(Likelihood):
    def __init__(self, C: torch.Tensor, R: torch.Tensor):
        self.C = torch.tensor(C, dtype=torch.float32)
        self.R = torch.tensor(R, dtype=torch.float32)
        assert self.C.shape[0] == self.R.shape[0] == self.R.shape[1]

    def __call__(self, state: torch.Tensor) -> dist.Distribution:
        return dist.MultivariateNormal(
            self.C * state,
            self.R
        )
