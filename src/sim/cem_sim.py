#!/usr/bin/env python3

import torch
from dataclasses import dataclass

from src.policy import Policy
from src.env import Env
from src.sim.base_sim import Simulation


@dataclass
class CEMSimulation(Simulation):
    def __init__(
        self,
        env: Env,
        policy: Policy,
        num_rounds: int,
        **kwargs,
    ):
        self.env = env
        self.policy = policy
        self.num_rounds = num_rounds

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
        return log_evidence
