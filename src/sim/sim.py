#!/usr/bin/env python3

import torch
from dataclasses import dataclass

from src.policy import Policy
from src.env import Env
from src.sim.resamplers import Resampler


@dataclass
class Simulation:
    def __init__(
        self,
        env: Env,
        policy: Policy,
        resampler: Resampler,
        num_rounds: int,
        **kwargs,
    ):
        self.env = env
        self.policy = policy
        self.resampler = resampler
        self.num_rounds = num_rounds
        self.log_num_particles = torch.log(torch.tensor(self.env.num_particles))

    def run(self):
        log_evidence = []
        for _ in range(self.num_rounds):
            obs, done = self.env.reset()
            log_evidence.append([torch.tensor([0.])])
            while not done.all():
                actions = self.policy.sample(obs, self.env.num_particles)
                obs, log_weights, done, _, _ = self.env.step(actions, update_state=False)
                log_evidence[-1].append(log_evidence[-1][-1] + torch.logsumexp(log_weights, axis=0) - self.log_num_particles)
                weights = log_weights.exp() * torch.logical_not(done)
                idx = self.resampler(weights, self.env.num_particles)
                self.env.set_resample_idx(idx)
                actions_hat = actions[idx]
                obs, _, done, _, _ = self.env.step(actions_hat, update_state=True)
                old_done = done.clone()
        return log_evidence
