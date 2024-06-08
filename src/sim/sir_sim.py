#!/usr/bin/env python3

import torch
from dataclasses import dataclass
from typing import List

from src.policy import Policy
from src.env import Env
from src.sim.resamplers import Resampler
from src.sim.base_sim import Simulation, Output


@dataclass
class SIROutput(Output):
    log_evidence: List[float]

    def __repr__(self):
        last_log_evidences = "\n".join(["\t\t%.2f" % x[-1] for x in self.log_evidence])
        return f'\nSIROutput(\n' \
               f'\tsuccess_rate={self.success_rate},\n' \
               f'\tnum_rounds={self.num_rounds},\n' \
               f'\ttime_limit={self.time_limit.item()},\n' \
               f'\tlog_evidence=\n{last_log_evidences}\n)'


@dataclass
class SIRSimulation(Simulation):
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

    def run(self) -> SIROutput:
        log_evidence = []
        num_successes = 0
        for _ in range(self.num_rounds):
            obs, _, done, truncated, _ = self.env.reset()
            log_evidence.append([torch.tensor([0.])])
            while not done.any() and not truncated.any():
                actions = self.policy.sample(obs, self.env.num_particles)
                obs, log_weights, _, _, _ = self.env.step(actions, update_state=False)
                log_evidence[-1].append(log_evidence[-1][-1] + torch.logsumexp(log_weights, axis=0) - self.log_num_particles)
                weights = log_weights.exp()
                # weights = (log_weights - torch.logsumexp(log_weights, axis=1)).exp()
                if weights.isnan().any():
                    import pdb; pdb.set_trace()
                idx = self.resampler(weights, self.env.num_particles)
                self.env.set_resample_idx(idx)
                actions_hat = actions[idx]
                obs, _, done, truncated, _ = self.env.step(actions_hat, update_state=True)
                num_successes += done.any()
        return SIROutput(
            success_rate=num_successes / self.num_rounds,
            num_rounds=self.num_rounds,
            time_limit=self.env.time_limit,
            log_evidence=log_evidence,
        )