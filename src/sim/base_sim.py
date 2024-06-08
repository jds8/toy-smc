#!/usr/bin/env python3

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.policy import Policy
from src.env import Env


@dataclass
class Output:
    success_rate: float
    num_rounds: int
    time_limit: int

    def __repr__(self):
        return f'Output(' \
               f'success_rate={self.success_rate}, ' \
               f'num_rounds={self.num_rounds}, ' \
               f'time_limit={self.time_limit})'


@dataclass
class Simulation:
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

    @abstractmethod
    def run(self) -> Output:
        pass
