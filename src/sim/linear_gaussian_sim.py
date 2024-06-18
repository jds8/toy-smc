#!/usr/bin/env python3

from typing import List

import torch

from src.policies.base_policy import Policy
from src.key_names.keys import Keys


class LinearGaussianSimulation:
    def __init__(
        self,
        env,
        proposal_policy: Policy,
        **kwargs,
    ):
        self.env = env
        self.proposal_policy = proposal_policy

    def run(self):
        while True:
            state, info = self.env.reset()
            done = info[Keys.DONE]
            num_particles = done.shape[0]
            truncated = done
            t = 0
            true_obs = torch.tensor([-2.0266,  0.8577,  2.9085, -0.2546,  3.8315, -0.1344, -1.3885,  0.3827,
                                     -2.1789, -1.7726])
            while not done.any() and not truncated.any():
                state, obs = state[:, :info[Keys.STATE_DIM]], state[:, info[Keys.STATE_DIM]:]
                obs = true_obs[t]
                print('mean: {}'.format(self.proposal_policy.mean))
                print('P: {}'.format(self.proposal_policy.P))
                actions, log_prob = self.proposal_policy.sample(
                    state,
                    t,
                    num_particles
                )
                action = torch.hstack([actions.squeeze(0), log_prob.reshape(-1, 1)]).numpy()
                state, log_weight, done, truncated, info = self.env.step(
                    action
                )
                t += 1
