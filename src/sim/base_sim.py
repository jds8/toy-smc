#!/usr/bin/env python3

from typing import List

from src.policies.base_policy import Policy


class Output:
    def __init__(self, log_evidences: List[float]):
        self.log_evidences = log_evidences

    def get_log_evidence_str(self):
        return "\n".join([
            "\t\t%.2f" % x[-1] for x in self.log_evidences
        ])

    def __repr__(self):
        last_log_evidences = self.get_log_evidence_str()
        return f'\nOutput(\n' \
               f'\tlog_evidence=\n{last_log_evidences}\n)'


class Simulation:
    def __init__(
        self,
        env,
        proposal_policy: Policy,
        **kwargs,
    ):
        self.env = env
        self.proposal_policy = proposal_policy

    def run(self) -> Output:
        raise NotImplementedError
