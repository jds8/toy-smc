#!/usr/bin/env python3

from typing import List

import torch

from src.policies.base_policy import Policy
from src.key_names.keys import Keys
from src.steppers.env_stepper import EnvStepper


class LinearGaussianSimulation:
    def __init__(
        self,
        env: EnvStepper,
        proposal_policy: Policy,
        **kwargs,
    ):
        self.env = env
        self.proposal_policy = proposal_policy
        self.start_time = 0
        self.time_increment = 1

    def run(self):
        while True:
            state, info = self.env.reset()
            self.proposal_policy.reset(info)
            done = info[Keys.DONE]
            num_particles = done.shape[0]
            truncated = done
            t = self.start_time
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
                t += self.time_increment


class ReverseSimulation(LinearGaussianSimulation):
    def __init__(
        self,
        env: EnvStepper,
        proposal_policy: Policy,
        **kwargs,
    ):
        super().__init__(env, proposal_policy)
        self.start_time = self.env.inner_env.time_limit - 1
        self.time_increment = -1
