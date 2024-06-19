#!/usr/bin/env python3

from typing import Tuple, Dict
from dataclasses import dataclass

import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.policies.base_policy import Policy
from src.likelihoods.likelihoods import Likelihood
from src.key_names.keys import Columns
from src.key_names.keys import Keys
from src.resamplers import Resampler


@dataclass
class StateSpaceEnv(gym.Env):
    def __init__(
        self,
        prior_policy: Policy,
        num_particles: int,
        dim: int,
        likelihood: Likelihood,
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
            shape=(num_particles, dim),
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
        self.dim = dim
        self.likelihood = likelihood
        self.states = torch.tensor(torch.nan).expand(self.num_particles, self.dim)
        self.trajectory = None
        # time_limit x 2 x dim where 2 is represents (state, obs)
        self.time_limit = torch.tensor([time_limit], dtype=int)
        self.ess_threshold = ess_threshold
        self.resampler = resampler
        self.train = train

        self.time = torch.tensor([0])
        self.done = torch.zeros(self.num_particles, 1, dtype=bool)

        self.seed = kwargs.seed if 'seed' in kwargs else 100
        torch.manual_seed(self.seed)

    def sample_likelihood(self, state: torch.Tensor) -> torch.Tensor:
        return self.likelihood(state).sample()

    def reward(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.likelihood(state).log_prob(obs)

    @staticmethod
    def get_info(
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
        dim,
    ):
        return {
            Keys.STATES: states,
            Keys.OBS: obs,
            Keys.NEXT_STATES: s_next,
            Keys.LOG_LIK: log_lik,
            Keys.LOG_PRIOR: log_prior,
            Keys.LOG_PROPOSAL: log_proposal,
            Keys.IDX: idx,
            Keys.RESAMPLED: resampled,
            Keys.DONE: done,
            Keys.GT_TRAJECTORY: gt_trajectory,
            Keys.STATE_DIM: dim,
        }

    def should_resample(self, weights) -> bool:
        ess = weights.sum()**2 / (weights**2).sum()
        return ess < self.ess_threshold * weights.shape[0]

    def get_next_obs(self):
        if self.done.any():
            return self.trajectory[-1, Columns.OBS_IDX]
        return self.trajectory[self.time, Columns.OBS_IDX]

    def step(self, actions: np.ndarray):
        action = actions[:, :-1]
        log_prob = actions[:, -1]

        assert self.action_space.contains(action), "Invalid action"

        next_states = torch.tensor(action)
        obs = self.trajectory[self.time, Columns.OBS_IDX]

        log_prior = self.prior_policy.log_prob(
            torch.tensor(action),
            self.time,
            self.states[:, :self.dim],
        )
        log_lik = self.reward(obs, next_states)
        log_proposal = torch.tensor(log_prob)
        log_weight = log_lik + log_prior - log_proposal
        weights = log_weight.exp()

        done = (self.time + 1 == self.time_limit).expand(next_states.shape)
        info = self.get_info(
            self.states,
            obs.expand(self.num_particles, self.dim),
            next_states,
            log_lik.reshape(self.num_particles, 1),
            log_prior.reshape(self.num_particles, 1),
            log_proposal.reshape(self.num_particles, 1),
            torch.arange(self.num_particles).reshape(-1, 1),
            torch.zeros(self.num_particles, 1, dtype=bool),
            done,
            self.trajectory,
            self.dim,
        )

        if self.should_resample(weights):
            idx = self.resampler(weights, self.num_particles)
            info = self.resample_step(idx, info)

        self.done = info[Keys.DONE]
        self.time += 1

        next_obs = self.get_next_obs().expand(
            self.num_particles,
            -1
        )
        self.states = torch.hstack([info[Keys.NEXT_STATES], next_obs])

        return (
            self.states,
            log_weight,
            info[Keys.DONE],
            info[Keys.DONE],
            info
        )

    def resample_step(
        self,
        resample_idx: torch.Tensor,
        info: Dict,
    ):
        info = self.get_info(
            info[Keys.STATES][resample_idx],
            info[Keys.OBS][resample_idx],
            info[Keys.NEXT_STATES][resample_idx],
            info[Keys.LOG_LIK][resample_idx],
            info[Keys.LOG_PRIOR][resample_idx],
            info[Keys.LOG_PROPOSAL][resample_idx],
            info[Keys.IDX][resample_idx],
            torch.ones(self.states.shape[0], 1, dtype=bool),
            info[Keys.DONE][resample_idx],
            info[Keys.GT_TRAJECTORY],
            self.dim
        )
        return info

    def generate_trajectory(self) -> torch.Tensor:
        """
        Generates a trajectory of observations for training.
        When testing, generates a single trajectory
        """
        if self.train or self.trajectory is None:
            state = torch.zeros(self.dim)
            trajectory = []
            for t in range(self.time_limit):
                state, _ = self.prior_policy.sample(state, t, 1)  # 1 x dim
                obs = self.sample_likelihood(state)  # 1 x dim
                # squeeze because sampling adds an extra batch dimension
                state = state.squeeze(0)
                obs = obs.squeeze(0)
                element = torch.stack([state, obs]).reshape(2, self.dim)
                trajectory.append(element)
            self.trajectory = torch.stack(trajectory)

    def reset(self, seed: int | None = None) -> Tuple[torch.Tensor, bool]:
        # resets the simulation and returns a new start state and done
        zeros = torch.zeros(self.num_particles, self.dim)
        nans = torch.tensor(torch.nan).expand(zeros.shape)
        self.generate_trajectory()
        obs = self.trajectory[0:1, Columns.OBS_IDX].expand(self.num_particles, self.dim)
        self.states = torch.hstack([zeros, obs])
        self.done = torch.zeros(self.num_particles, 1, dtype=bool)
        self.time = torch.tensor([0])
        info = self.get_info(
            self.states,
            self.trajectory[:, Columns.OBS_IDX],
            nans,
            nans,
            nans,
            nans,
            nans,
            nans,
            self.done,
            self.trajectory,
            self.dim,
        )
        return self.states, info
