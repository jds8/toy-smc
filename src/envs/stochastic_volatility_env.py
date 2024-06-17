#!/usr/bin/env python3

from collections import namedtuple
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import torch
import torch.distributions as dist
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.policies.stochastic_volatility_policies import SVPrior
from src.key_names.keys import Columns as SVCols
from src.key_names.keys import Keys as SVKeys
from src.resamplers import Resampler


@dataclass
class SVEnv(gym.Env):
    def __init__(
        self,
        prior_policy: SVPrior,
        num_particles: int,
        num_currencies: int,
        beta: float,
        time_limit: int,
        ess_threshold: int,
        resampler: Resampler,
        train: bool,
        **kwargs,
    ):
        super().__init__()

        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_particles, num_currencies),
            dtype=float
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_particles, 2),
            dtype=np.float32
        )

        self.prior_policy = prior_policy
        self.num_particles = num_particles
        self.num_currencies = num_currencies
        self.beta = beta
        self.states = torch.tensor(torch.nan).expand(self.num_particles, self.num_currencies)
        # time_limit x 2 x num_currencies where 2 is represents (state, obs)
        self.trajectory = None
        self.time_limit = torch.tensor([time_limit], dtype=int)
        self.ess_threshold = ess_threshold
        self.resampler = resampler
        self.train = train

        self.time = torch.tensor([0])
        self.done = torch.zeros(self.num_particles, 1, dtype=bool)

        self.seed = kwargs.seed if 'seed' in kwargs else 100
        torch.manual_seed(self.seed)

    def likelihood(self, state: torch.Tensor) -> torch.Tensor:
        cov = self.beta * torch.exp(state / 2.)
        return dist.Independent(
            dist.Normal(torch.zeros_like(state), cov),
            reinterpreted_batch_ndims=1
        )

    def sample_likelihood(self, state: torch.Tensor) -> torch.Tensor:
        return self.likelihood(state).sample()

    def reward(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.likelihood(state).log_prob(obs)

    def get_info(
        self,
        states,
        obs,
        s_next,
        log_lik,
        log_prior,
        log_proposal,
        idx,
        resampled,
        done,
        gt_trajectory,
    ):
        return {
            SVKeys.STATES: states,
            SVKeys.OBS: obs,
            SVKeys.NEXT_STATES: s_next,
            SVKeys.LOG_LIK: log_lik,
            SVKeys.LOG_PRIOR: log_prior,
            SVKeys.LOG_PROPOSAL: log_proposal,
            SVKeys.IDX: idx,
            SVKeys.RESAMPLED: resampled,
            SVKeys.DONE: done,
            SVKeys.GT_TRAJECTORY: gt_trajectory,
        }

    def should_resample(self, weights) -> bool:
        ess = weights.sum()**2 / (weights**2).sum()
        return ess < self.ess_threshold * weights.shape[0]

    def step(self, actions: np.ndarray):
        action = actions[:, :-1]
        log_prob = actions[:, -1]

        assert self.action_space.contains(action), "Invalid action"

        next_states = torch.tensor(action)
        obs = self.trajectory[self.time, SVCols.OBS_IDX]

        log_prior = self.prior_policy.log_prob(
            torch.tensor(action),
            self.time,
            self.states
        )
        log_lik = self.reward(obs, next_states)
        log_proposal = torch.tensor(log_prob)
        log_weight = log_lik + log_prior - log_proposal
        weights = log_weight.exp()

        done = (self.time + 1 == self.time_limit).expand(next_states.shape)
        info = self.get_info(
            self.states,
            obs.expand(self.num_particles, self.num_currencies),
            next_states,
            log_lik.reshape(self.num_particles, 1),
            log_prior.reshape(self.num_particles, 1),
            log_proposal.reshape(self.num_particles, 1),
            torch.arange(self.num_particles).reshape(-1, 1),
            torch.zeros(self.num_particles, 1, dtype=bool),
            done,
            self.trajectory,
        )

        if self.should_resample(weights):
            idx = self.resampler(weights, self.num_particles)
            info = self.resample_step(idx, info)

        self.states = info[SVKeys.NEXT_STATES]
        self.done = info[SVKeys.DONE]
        self.time += 1
        return (
            info[SVKeys.NEXT_STATES],
            log_weight,
            info[SVKeys.DONE],
            info[SVKeys.DONE],
            info
        )

    def resample_step(
        self,
        resample_idx: torch.Tensor,
        info: Dict,
    ):
        info = self.get_info(
            info[SVKeys.STATES][resample_idx],
            info[SVKeys.OBS][resample_idx],
            info[SVKeys.NEXT_STATES][resample_idx],
            info[SVKeys.LOG_LIK][resample_idx],
            info[SVKeys.LOG_PRIOR][resample_idx],
            info[SVKeys.LOG_PROPOSAL][resample_idx],
            info[SVKeys.IDX][resample_idx],
            torch.ones(self.states.shape[0], 1, dtype=bool),
            info[SVKeys.DONE][resample_idx],
            info[SVKeys.GT_TRAJECTORY],
        )
        return info

    def generate_trajectory(self) -> torch.Tensor:
        """
        Generates a trajectory of observations for training.
        When testing, generates a single trajectory
        """
        if self.train or self.trajectory is None:
            state = torch.nan
            trajectory = []
            for t in range(self.time_limit):
                state, _ = self.prior_policy.sample(state, t, 1)  # 1 x num_currencies
                obs = self.sample_likelihood(state)  # 1 x num_currencies
                # squeeze because sampling adds an extra batch dimension
                state = state.squeeze(0)
                obs = obs.squeeze(0)
                element = torch.stack([state, obs]).reshape(2, self.num_currencies)
                trajectory.append(element)
            self.trajectory = torch.stack(trajectory)

    def reset(self, seed: int | None = None) -> Tuple[torch.Tensor, bool]:
        # resets the simulation and returns a new start state and done
        nans = torch.tensor(torch.nan).expand(self.num_particles, self.num_currencies)
        self.states = nans
        self.generate_trajectory()
        self.done = torch.zeros(self.num_particles, 1, dtype=bool)
        self.time = torch.tensor([0])
        info = self.get_info(
            self.states,
            self.trajectory[:, SVCols.OBS_IDX],
            nans,
            nans,
            nans,
            nans,
            nans,
            nans,
            self.done,
            self.trajectory,
        )
        return self.states, info
