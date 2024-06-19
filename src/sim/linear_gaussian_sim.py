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
            self.proposal_policy.reset(info)
            done = info[Keys.DONE]
            num_particles = done.shape[0]
            truncated = done
            t = 0
            while not done.any() and not truncated.any():
                actions, log_prob = self.proposal_policy.sample(
                    state,
                    t,
                    num_particles
                )
                action = torch.hstack([actions.squeeze(-1), log_prob.reshape(-1, 1)]).numpy()
                state, log_weight, done, truncated, info = self.env.step(
                    action
                )
                t += 1
