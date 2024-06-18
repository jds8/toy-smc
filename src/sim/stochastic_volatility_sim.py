#!/usr/bin/env python3

import torch
from dataclasses import dataclass
from typing import List

from src.key_names.keys import Keys as SVKeys


class SVSimulation:
    def run(self):
        while True:
            state, info = self.env.reset()
            done = info[SVKeys.DONE]
            num_particles = done.shape[0]
            truncated = done
            t = 0
            while not done.any() and not truncated.any():
                actions, log_prob = self.proposal_policy.sample(
                    state,
                    t,
                    num_particles
                )
                action = torch.hstack([actions, log_prob.reshape(-1, 1)]).numpy()
                state, log_weight, done, truncated, info = self.env.step(
                    action
                )
                t += 1
