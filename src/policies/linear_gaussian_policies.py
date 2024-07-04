#!/usr/bin/env python3

from typing import Union, List, Dict, Optional

import torch
import torch.distributions as dist

from src.policies.base_policy import Policy
from src.key_names.keys import Keys, Columns


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
        new_P = torch.mm(paren_term, prior_covariance)
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
        self.reset()

    def _predict(self):
        x_prior = torch.mm(self.A, self.means[-1])
        p_prior = torch.mm(torch.mm(self.A, self.Ps[-1]), self.A.T) + self.Q
        return x_prior, p_prior

    def _correct(self, p_prior, x_prior, obs):
        mat = torch.mm(torch.mm(self.C, p_prior), self.C.T) + self.R
        mat_inv = mat.pinverse()
        k_n = torch.mm(torch.mm(p_prior, self.C.T), mat_inv)
        new_mean = x_prior + torch.mm(k_n, obs[:, :self.state_dim] - torch.mm(self.C, x_prior))
        paren_term = torch.eye(p_prior.shape[0]) - torch.mm(k_n, self.C)
        sum_term = torch.mm(torch.mm(k_n, self.R), k_n.T)
        new_P = torch.mm(torch.mm(paren_term, p_prior), paren_term.T) + sum_term
        return new_mean, new_P

    def _kalman_update(self, obs, t):
        """
        Pilfered from https://www.kalmanfilter.net/multiSummary.html
        """
        if t == 0:
            x_prior = torch.zeros(self.A.shape[0], 1)
            p_prior = self.Q
        else:
            x_prior, p_prior = self._predict()
        new_mean, new_P = self._correct(p_prior, x_prior, obs)
        self.means.append(new_mean)
        self.Ps.append(new_P)

    def compute_dist(self, obs: torch.Tensor, t: int):
        """
        Computes p(x_t|y_{0:t})
        """
        assert self.t == t, f"Kalman filter has been updated {self.t} "\
                            f"times but attempting to compute dist " \
                            f"for the {t}th iteration"
        self._kalman_update(obs, t)
        self.t += 1
        return dist.MultivariateNormal(self.means[-1], self.Ps[-1])

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

    def reset(self, _: Optional[Dict]=None):
        self.means = []
        self.Ps = []
        self.t = 0


class OptimalSmoothingPolicy(Policy):
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
        self.kfilter = MarginalFilteringPolicy(A, C, Q, R)
        self.means = []
        self.Ps = []
        self.ys = None
        self.n = 0

    def pre_filter(self):
        for n, y in enumerate(self.ys):
            self.kfilter.compute_dist(y.reshape(-1, 1), n)

    def _smooth(self, x_next_obs: torch.Tensor, n: int):
        """
        See equation (13) of Kitagawa - 2023 - Revisiting the Two-Filter Formula
        """
        x_next = x_next_obs[:, :self.state_dim].reshape(-1, *self.A.shape)
        Vnn = self.kfilter.Ps[n]
        Q_inv = self.Q.pinverse()
        precision = Vnn.pinverse() + self.A.T @ Q_inv @ self.A
        new_P = precision.pinverse()

        xnn = self.kfilter.means[n]
        new_x = new_P @ (precision @ xnn + self.A.T @ Q_inv @ x_next)

        self.means = [new_x] + self.means
        self.Ps = [new_P] + self.Ps

    def compute_dist(self, x_next_obs: torch.Tensor, n: int):
        """
        Computes p(x_t|y_{0:t})
        """
        assert self.n == n, f"Kalman smoother has been updated {len(self.ys)-1-self.n} "\
                            f"times but attempting to compute dist " \
                            f"for the N-n={len(self.ys)-1-n}th iteration"
        if len(self.ys) > self.n + 1:
            self._smooth(x_next_obs, n)
        self.n -= 1
        return dist.MultivariateNormal(self.means[0], self.Ps[0])

    def sample(
        self,
        obs: torch.Tensor,
        n: int,
        num_particles: int
    ) -> torch.Tensor:
        gauss_dist = self.compute_dist(obs, n)
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

    def reset(self, info: Optional[Dict] = None):
        self.ys = info[Keys.GT_TRAJECTORY][:, Columns.OBS_IDX]
        num_ys = len(self.ys)
        self.n = num_ys - 1
        self.kfilter.reset(info)
        self.pre_filter()
        batch_size = info[Keys.STATES].shape[0]
        self.means = [self.kfilter.means[-1].tile(batch_size, 1, 1)]
        self.Ps = [self.kfilter.Ps[-1]]


class KalmanSmoother(Policy):
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
        self.kfilter = MarginalFilteringPolicy(A, C, Q, R)
        self.means = []
        self.Ps = []
        self.ys = None
        self.n = 0

    def pre_filter(self):
        for n, y in enumerate(self.ys):
            self.kfilter.compute_dist(y.reshape(-1, 1), n)

    def _smooth(self, n: int):
        """
        See equation (13) of Kitagawa - 2023 - Revisiting the Two-Filter Formula
        """
        Vnn = self.kfilter.Ps[n]
        Vn1n = torch.mm(self.A, torch.mm(Vnn, self.A.T)) + self.Q
        An = torch.mm(Vnn, torch.mm(self.A.T, Vn1n.pinverse()))
        xnn = self.kfilter.means[n]
        xnN = xnn + torch.mm(An, self.means[n+1] - torch.mm(self.A, xnn))
        VnN = Vnn + torch.mm(An, torch.mm(self.Ps[n+1] - Vn1n, An.T))
        self.means[n] = xnN
        self.Ps[n] = VnN

    def compute_dist(self, _: torch.Tensor, n: int):
        """
        Computes p(x_t|y_{0:t})
        """
        assert self.n == n, f"Kalman smoother has been updated {len(self.ys)-1-self.n} "\
                            f"times but attempting to compute dist " \
                            f"for the N-n={len(self.ys)-1-n}th iteration"
        self._smooth(n)
        self.n -= 1
        return dist.MultivariateNormal(self.means[-1], self.Ps[-1])

    def sample(
        self,
        obs: torch.Tensor,
        n: int,
        num_particles: int
    ) -> torch.Tensor:
        gauss_dist = self.compute_dist(obs, n)
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

    def reset(self, info: Optional[Dict] = None):
        self.kfilter = MarginalFilteringPolicy(
            self.A,
            self.C,
            self.Q,
            self.R
        )
        self.ys = info[Keys.GT_TRAJECTORY][:, Columns.OBS_IDX]
        num_ys = len(self.ys)
        self.n = num_ys - 2
        self.kfilter.reset(info)
        self.pre_filter()
        first_mean = self.kfilter.means[-1]
        self.means = torch.zeros(num_ys, *first_mean.shape)
        self.means[-1] = first_mean
        first_cov = self.kfilter.Ps[-1]
        self.Ps = torch.zeros(num_ys, *first_cov.shape)
        self.Ps[-1] = first_cov
