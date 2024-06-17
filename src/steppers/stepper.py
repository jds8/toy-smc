#!/usr/bin/env python3

from abc import ABC
import torch
from typing import Tuple, Dict


class Stepper(ABC):
    def pre_step(self, state: torch.Tensor, action: torch.Tensor):
        pass

    def post_step(self, action: torch.Tensor, env_out: Tuple, recorder_out: Tuple):
        pass

    def pre_reset(self):
        pass

    def post_reset(self, out: Tuple, data_path_name: str):
        pass

    def pre_close(self):
        pass
