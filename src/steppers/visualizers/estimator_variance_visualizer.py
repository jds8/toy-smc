#!/usr/bin/env python3

import os

import torch
import numpy as np

from hydra.core.hydra_config import HydraConfig

from src.key_names.keys import Keys, VizKeys
from src.steppers.visualizers.base_visualizer import BaseVisualizer
from src.file_names.file_names import Data


class EstimatorVarianceVisualizer(BaseVisualizer):
    def __init__(
        self,
        name: str,
        title: str,
        **kwargs
    ):
        data_path = '{}_{}'.format(name, 0)
        super().__init__(name, data_path, title)

    def load_data(self, data_path: str):
        data = torch.load(data_path)
        self.gt_data = data[f'{Keys.GT_TRAJECTORY}_0'].squeeze().numpy()
        self.mean_data = data[f'{VizKeys.STATE_MEAN}'].squeeze().numpy()
        self.upper_ci_data = data[f'{VizKeys.STATE_UPPER_STD}'].squeeze().numpy()
        self.lower_ci_data = data[f'{VizKeys.STATE_LOWER_STD}'].squeeze().numpy()
        self.time_limit = self.mean_data.shape[0]
        self.times = data[VizKeys.TIME].numpy()

    def make_plot(self, data_path):
        self.load_data(data_path)
        super().make_plot()

    def post_reset(self, _, estimator_stats):
        data_path = '{}/{}_{}.pt'.format(
            HydraConfig.get().run.dir,
            Data.STATE_DATA,
            self.num_resets
        )
        if os.path.exists(data_path):
            self.make_plot(data_path)
        self.num_resets += 1
