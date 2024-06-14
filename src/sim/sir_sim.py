#!/usr/bin/env python3

import torch
from dataclasses import dataclass
from typing import List

import src.policy as policy
from src.env import Env, RecorderEnv
from src.sim.resamplers import Resampler
from src.sim.base_sim import Simulation, Output


@dataclass
class SIROutput(Output):
    log_evidences: List[float]

    def __repr__(self):
        last_log_evidences = "\n".join([
            "\t\t%.2f" % x[-1] for x in self.log_evidences
        ])
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
        proposal_policy: policy.Policy,
        resampler: Resampler,
        num_rounds: int,
        ess_threshold: float,
        **kwargs,
    ):
        self.env = env
        self.proposal_policy = proposal_policy
        self.resampler = resampler
        self.num_rounds = num_rounds
        self.ess_threshold = ess_threshold
        self.num_particles = self.env.get_num_particles()
        self.log_num_particles = torch.log(torch.tensor(self.num_particles))

    def run(self) -> SIROutput:
        log_evidences = []
        num_successes = 0
        for _ in range(self.num_rounds):
            done = torch.zeros(self.num_particles, dtype=bool)
            truncated = done
            obs, info = self.env.reset()
            log_w = torch.tensor([0.])
            log_evidences.append([log_w])
            while not done.any() and not truncated.any():
                actions, log_prob = self.proposal_policy.sample(
                    obs,
                    self.num_particles
                )
                obs, log_joint, done, truncated, info = self.env.step(
                    actions.numpy(),
                    update_state=False
                )
                log_weights = log_joint - log_prob
                log_ev = torch.logsumexp(
                    log_weights,
                    axis=0
                ) - self.log_num_particles
                log_evidences[-1].append(log_w + log_ev)
                weights = log_weights.exp()
                if isinstance(self.env, RecorderEnv):
                    # self.env.store_weights(torch.tensor([1.]))
                    self.env.store_weights(weights / weights.sum())
                if self.should_resample(weights):
                    idx = self.resampler(weights, self.num_particles)
                    obs, _, done, truncated, _ = self.env.resample_step(
                        actions.numpy(),
                        idx
                    )
                    log_w = torch.tensor([0.])
                else:
                    self.env.set_from_info(info)
                    log_w = log_evidences[-1][-1]
                num_successes += done.any()
        return SIROutput(
            success_rate=num_successes / self.num_rounds,
            num_rounds=self.num_rounds,
            time_limit=self.env.get_time_limit(),
            log_evidences=log_evidences,
        )

    def should_resample(self, weights) -> bool:
        # return False
        ess = weights.sum()**2 / (weights**2).sum()
        return ess < self.ess_threshold * weights.shape[0]
