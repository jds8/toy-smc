#!/usr/bin/env python3

import torch

from src.steppers.recorders.recorder import Recorder
from src.key_names.keys import Keys, VizKeys


class ReverseRecorder(Recorder):
    def __init__(
        self,
        num_particles: int,
        **kwargs
    ):
        super().__init__(num_particles)

    def save(self, combined_info: dict, data_path_name: str):
        new_info = combined_info.copy()
        # Reverse the times because a backward smoother was used
        time = torch.arange(new_info[f'{VizKeys.STATE_MEAN}'].shape[0]-1, -1, -1)
        new_info[VizKeys.TIME] = time
        new_info[f'{Keys.GT_TRAJECTORY}_0'] = new_info[f'{Keys.GT_TRAJECTORY}_0'].flip(dims=(0,))
        # Ground truth data is the only data that is not reversed, so reverse it
        torch.save(new_info, data_path_name)
