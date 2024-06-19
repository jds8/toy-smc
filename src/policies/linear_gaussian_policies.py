#!/usr/bin/env python3

from typing import Union, List, Dict, Optional

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
        action = gauss_dist.rsample()
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


class OptimalFilteringPolicy(Policy):
    """
    Computes p(x_t|y_t, x_{t-1})
    """
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
        self.state_dim = self.A.shape[0]
        self.obs_dim = self.C.shape[0]
        self.reset()

    def _correct(self, x_prev_obs):
        """
        x_prev is the known previous state x_{t-1}
        obs is the known current observation y_t
        in the comments below, v_t ~ N(0, R) is the observation noise
        """
        x_prev = x_prev_obs[:, :self.state_dim].reshape(-1, *self.A.shape)
        obs = x_prev_obs[:, self.state_dim:].reshape(-1, *self.C.shape)
        batch_dim = x_prev.shape[0]
        batch_A = self.A.tile(batch_dim, 1, 1)
        batch_C = self.C.tile(batch_dim, 1, 1)
        # x_t = Ax_{t-1} + N(0, Q)
        # Var(x_t) = Var(Ax_{t-1}) + Q = AVar(x_{t-1})A^T + Q
        prior_predictive_mean = torch.bmm(batch_A, x_prev)  # E[x_t|x_{t-1}]
        prior_covariance = self.Q  # Var[x_t|x_{t-1}]
        predicted_obs = torch.bmm(batch_C, prior_predictive_mean)  # E[y_t|x_{t-1}]
        residual = obs - predicted_obs  # obs - E[y_t|x_{t-1}]
        # Var[obs - y_t|x_{t-1}] = Var[y_t|x_{t-1}] since obs is constant/known
        # S = Var[y_t|x_{t-1}] = Var[Cx_t + v_t|x_{t-1}] = CVar[x_t|x_{t-1}]C^T + R
        residual_covariance = torch.mm(torch.mm(self.C, prior_covariance), self.C.T) + self.R
        # gain matrix is prior_covariance * C * S^{-1}
        # gain is derived to minimize MSE/trace of covariance
        k_n = torch.mm(torch.mm(prior_covariance, self.C.T), residual_covariance.pinverse())
        batch_k_n = k_n.tile(batch_dim, 1, 1)
        new_mean = prior_predictive_mean + torch.bmm(batch_k_n, residual)
        paren_term = torch.eye(k_n.shape[0]) - torch.mm(k_n, self.C)
        new_P = torch.mm(paren_term, self.Q)
        return new_mean, new_P

    def _kalman_update(self, x_prev_obs: torch.Tensor):
        """
        Pilfered from https://www.kalmanfilter.net/multiSummary.html
        but where x_prev is provided in xy[:self.state_dim]
        """
        new_mean, new_P = self._correct(x_prev_obs)
        self.means.append(new_mean)
        self.Ps.append(new_P)

    def compute_dist(self, obs: torch.Tensor, t: int):
        """
        Computes p(x_t|y_{0:t},x_{t-1})
        """
        self._kalman_update(obs)
        return dist.MultivariateNormal(self.means[-1], self.Ps[-1])

    def reset(self, _: Optional[Dict]=None):
        self.means = []
        self.Ps = []

    def sample(
        self,
        obs: torch.Tensor,
        t: int,
        num_particles: int
    ) -> torch.Tensor:
        gauss_dist = self.compute_dist(obs, t)
        action = gauss_dist.rsample()
        log_prob = gauss_dist.log_prob(action)
        return action, log_prob

    def log_prob(
        self,
        action: torch.Tensor,
        t: int,
        _: torch.Tensor
    ) -> torch.Tensor:
        gauss_dist = dist.MultivariateNormal(self.means[t], self.Ps[t])
        return gauss_dist.log_prob(action)


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


class MarginalFilteringPolicy(Policy):
    """
    This class computes the marginal filtering posterior:
    p(x_t:y_{1:t}).
    """
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
        self.state_dim = self.A.shape[0]
        self.obs_dim = self.C.shape[0]
        self.mean = torch.tensor([[-2.9888e-01]])  # torch.zeros(self.A.shape[0], 1)
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
        new_mean = x_next + torch.mm(k_n, obs[:, self.state_dim:] - torch.mm(self.C, x_next))
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
