#!/usr/bin/inner_env python3


from abc import ABC
from typing import Sequence, Tuple, Dict, Optional
import torch
import gymnasium as gym

from src.steppers.stepper import Stepper
from src.steppers.recorders.recorder import Recorder


class EnvStepper(Stepper):
    def __init__(
        self,
        inner_env: gym.Env,
        recorder: Recorder,
        steppers: Optional[Sequence[Stepper]] = [],
        *args,
        **kwargs
    ):
        self.inner_env = inner_env
        self.recorder = recorder
        self.steppers = steppers

    def pre_step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ):
        for stepper in self.steppers:
            stepper.pre_step(state, action)

    def post_step(self, action: torch.Tensor, env_out: Tuple):
        recorder_out = self.recorder.post_step(action, env_out)
        for stepper in self.steppers:
            stepper.post_step(action, env_out, recorder_out)

    def step(self, action: torch.Tensor):
        state = self.inner_env.states
        self.pre_step(state, action)
        env_out = self.inner_env.step(action)
        self.post_step(action, env_out)
        return env_out

    def pre_reset(self):
        for stepper in self.steppers:
            stepper.pre_reset()

    def post_reset(self, out: Tuple):
        estimator_stats = self.recorder.post_reset(out)
        for stepper in self.steppers:
            stepper.post_reset(out, estimator_stats)

    def reset(self):
        self.pre_reset()
        out = self.inner_env.reset()
        self.post_reset(out)
        return out

    def close(self):
        data_path_name = self.recorder.pre_close()
        for stepper in self.steppers:
            stepper.pre_close(data_path_name)
        self.inner_env.close()
