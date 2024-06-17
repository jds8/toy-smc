#!/usr/bin/env python3

import torch
from dataclasses import dataclass
from typing import List

from src.policies.base_policy import Policy
from src.envs.toy_control_env import ToyControlEnv, RecorderEnv
from src.resamplers import Resampler
from src.sim.base_sim import Output, Simulation


class SIROutput(Output):
    success_rate: float
    num_rounds: int

    def __repr__(self):
        last_log_evidences = self.get_log_evidence_str()
        return f'\nSIROutput(\n' \
               f'\tsuccess_rate={self.success_rate},\n' \
               f'\tnum_rounds={self.num_rounds},\n' \
               f'\ttime_limit={self.time_limit},\n' \
               f'\tlog_evidence=\n{last_log_evidences}\n)'


class SIRSimulation(Simulation):
    def __init__(
        self,
        env: ToyControlEnv,
        proposal_policy: Policy,
        resampler: Resampler,
        num_rounds: int,
        ess_threshold: float,
        **kwargs,
    ):
        super().__init__(
            env,
            proposal_policy,
            resampler,
            ess_threshold
        )
        self.num_rounds = num_rounds

    def run(self) -> SIROutput:
        log_evidences = []
        num_successes = 0
        for _ in range(self.num_rounds):
            done = torch.zeros(self.num_particles, dtype=bool)
            truncated = done
            obs, info = self.env.reset()
            log_w = torch.tensor([0.])
            log_evidences.append([log_w])
            t = 0
            while not done.any() and not truncated.any():
                actions, log_prob = self.proposal_policy.sample(
                    obs,
                    t,
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
                t += 1
        return SIROutput(
            time_limit=self.env.get_time_limit(),
            log_evidences=log_evidences,
            success_rate=num_successes / self.num_rounds,
            num_rounds=self.num_rounds,
        )
