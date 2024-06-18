#!/usr/bin/env python3

from typing import Union, List

import torch
import torch.distributions as dist

from src.policies.base_policy import Policy


class GaussianPolicy(Policy):
    def __init__(
        self,
        A: Union[torch.Tensor, List],
        Q: Union[torch.Tensor, List],
        **kwargs
    ):
        self.A = torch.tensor(A, dtype=torch.float32)
        self.Q = torch.tensor(Q, dtype=torch.float32)
        assert self.A.shape[0] == self.Q.shape[0] == self.Q.shape[1]
        super().__init__(dim=self.A.shape[0])

    def compute_dist(self, obs: torch.Tensor, t: int):
        mean = self.A * obs if self.A > 0 else torch.zeros(self.dim)
        return dist.MultivariateNormal(mean, self.Q)

    def sample(
        self,
        obs: torch.Tensor,
        t: int,
        num_particles: int
    ) -> torch.Tensor:
        gauss_dist = self.compute_dist(obs, t)
        action = gauss_dist.rsample([num_particles])
        log_prob = gauss_dist.log_prob(action)
        return action, log_prob

    def log_prob(
        self,
        action: torch.Tensor,
        t: int,
        obs: torch.Tensor
    ) -> torch.Tensor:
        gauss_dist = self.compute_dist(obs, t)
        return gauss_dist.log_prob(action)


class FilteringPolicy(Policy):
    def __init__(
        self,
        A: Union[torch.Tensor, List],
        C: Union[torch.Tensor, List],
        Q: Union[torch.Tensor, List],
        R: Union[torch.Tensor, List],
    ):
        self.A = torch.tensor(A)
        self.C = torch.tensor(C)
        self.Q = torch.tensor(Q)
        self.R = torch.tensor(R)
        self.mean = torch.zeros(self.A.shape[0], 1)
        self.P = torch.eye(self.A.shape[0])
        self.t = 0

    def _predict(self):
        x_next = torch.mm(self.A, self.mean)
        p_next = torch.mm(torch.mm(self.A, self.P), self.A.T) + self.Q
        return x_next, p_next

    def _correct(self, p_next, x_next, obs):
        mat = torch.mm(torch.mm(self.C, p_next), self.C.T) + self.R
        mat_inv = mat.pinverse()
        k_n = torch.mm(torch.mm(p_next, self.C.T), mat_inv)
        new_mean = x_next + torch.mm(k_n, obs - torch.mm(self.C, x_next))
        paren_term = torch.eye(p_next.shape[0]) - torch.mm(k_n, self.C)
        sum_term = torch.mm(torch.mm(k_n, self.R), k_n.T)
        new_P = torch.mm(torch.mm(paren_term, p_next), paren_term.T) + sum_term
        return new_mean, new_P

    def _kalman_update(self, obs):
        """
        Pilfered from https://www.kalmanfilter.net/multiSummary.html
        """
        x_next, p_next = self._predict()
        new_mean, new_P = self._correct(p_next, x_next, obs)
        self.mean = new_mean
        self.P = new_P


class SmoothingPolicy(Policy):
    def __init__(self, ys):
        self.ys = ys

    def compute_dist(self, obs: torch.Tensor, t: int):
        mean = self.A * obs if self.A > 0 else torch.zeros(self.dim)
        return dist.MultivariateNormal(mean, self.Q)

    def sample(
        self,
        obs: torch.Tensor,
        t: int,
        num_particles: int
    ) -> torch.Tensor:
        gauss_dist = self.compute_dist(obs, t)
        action = gauss_dist.rsample([num_particles])
        log_prob = gauss_dist.log_prob(action)
        return action, log_prob

    def log_prob(
        self,
        action: torch.Tensor,
        t: int,
        obs: torch.Tensor
    ) -> torch.Tensor:
        gauss_dist = self.compute_dist(obs, t)
        return gauss_dist.log_prob(action)


class FilteringDistribution(Policy):
    def __init__(
        self,
        A: Union[torch.Tensor, List],
        C: Union[torch.Tensor, List],
        Q: Union[torch.Tensor, List],
        R: Union[torch.Tensor, List],
    ):
        self.A = torch.tensor(A)
        self.C = torch.tensor(C)
        self.Q = torch.tensor(Q)
        self.R = torch.tensor(R)
        self.mean = torch.tensor([[0.2425]]) #torch.zeros(self.A.shape[0], 1)
        self.P = self.Q
        self.t = 0

    def _predict(self):
        x_next = torch.mm(self.A, self.mean)
        p_next = torch.mm(torch.mm(self.A, self.P), self.A.T) + self.Q
        return x_next, p_next

    def _correct(self, p_next, x_next, obs):
        mat = torch.mm(torch.mm(self.C, p_next), self.C.T) + self.R
        mat_inv = mat.pinverse()
        k_n = torch.mm(torch.mm(p_next, self.C.T), mat_inv)
        new_mean = x_next + torch.mm(k_n, obs - torch.mm(self.C, x_next))
        paren_term = torch.eye(p_next.shape[0]) - torch.mm(k_n, self.C)
        sum_term = torch.mm(torch.mm(k_n, self.R), k_n.T)
        new_P = torch.mm(torch.mm(paren_term, p_next), paren_term.T) + sum_term
        return new_mean, new_P

    def _kalman_update(self, obs):
        """
        Pilfered from https://www.kalmanfilter.net/multiSummary.html
        """
        x_next, p_next = self._predict()
        new_mean, new_P = self._correct(p_next, x_next, obs)
        self.mean = new_mean
        self.P = new_P

    def compute_dist(self, obs: torch.Tensor, t: int):
        """
        Computes p(x_t|y_{0:t})
        """
        assert self.t == t, f"Kalman Filter has been updated {self.t} "\
                            f"times but attempting to compute dist " \
                            f"for the {t}th iteration"
        self._kalman_update(obs)
        self.t += 1
        return dist.MultivariateNormal(self.mean, self.P)

    def sample(
        self,
        obs: torch.Tensor,
        t: int,
        num_particles: int
    ) -> torch.Tensor:
        gauss_dist = self.compute_dist(obs, t)
        action = gauss_dist.rsample([num_particles])
        log_prob = gauss_dist.log_prob(action)
        return action, log_prob

    def log_prob(
        self,
        action: torch.Tensor,
        t: int,
        obs: torch.Tensor
    ) -> torch.Tensor:
        gauss_dist = self.compute_dist(obs, t)
        return gauss_dist.log_prob(action)
