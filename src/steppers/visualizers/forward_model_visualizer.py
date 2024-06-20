#!/usr/bin/env python3

from abc import abstractmethod
from typing import Tuple, Dict

import torch
import numpy as np

from hydra.core.hydra_config import HydraConfig

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from IPython.display import display, clear_output

from src.key_names.keys import Keys, Columns, VizKeys
from src.steppers.stepper import Stepper


class Visualizer(Stepper):
    def __init__(self, name: str, data_path:str, **kwargs):
        self.name = name
        self.data_path = data_path

        self.min_state = None
        self.max_state = None
        self.min_y = None
        self.max_y = None
        self.fig = None
        self.gt_state = None
        self.gt_obs = None

    def load(self):
        dataset = torch.load(self.data_path)
        dataset[VizKeys.GT_TRAJECTORY]

    def initialize_plot(
        self,
        gt_trajectory: torch.Tensor,
    ):
        clear_output(wait=True)

        # Create initial figure
        self.fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        x_vals = np.arange(self.time_limit) + 1
        self.gt_state = go.Scatter(
            x=x_vals,
            y=gt_trajectory[:, Columns.STATE_IDX].numpy(),
            mode='lines',
            name='Simulated Sequence (Ground Truth)',
            marker=dict(color='blue', size=1),
        )
        self.gt_obs = go.Scatter(
            x=x_vals,
            y=gt_trajectory[:, Columns.OBS_IDX].numpy(),
            mode='markers',
            name='Simulated Sequence (Ground Truth)',
            marker=dict(color='red', symbol='diamond', size=3),
        )
        self.fig.add_trace(self.gt_state)
        self.fig.add_trace(self.gt_obs)

        # set y-axis limits
        self.fig.update_xaxes(title_text='Time', range=[0, 500])

        # set y-axis limits
        self.fig.update_yaxes(range=[self.min_y, self.max_y])

        # update display
        self.fig.update_traces()
        display(self.fig, display_id=True)

    def save(self):
        if self.fig:
            img_name = '{}/{}_{}.pdf'.format(
                HydraConfig.get().run.dir,
                self.name,
                self.num_resets
            )
            self.fig.write_image(img_name, format='pdf')

    def plot(self):
        # save old figure
        self.save()

        # update new figure
        _, info = out

        gt_trajectory = info[Keys.GT_TRAJECTORY]
        # in order to visualize observations,
        # they need to be one dimensional scalars
        assert gt_trajectory.shape == (self.time_limit, 2, 1)

        gt_trajectory = gt_trajectory.squeeze(-1)

        self.min_state = gt_trajectory[:, Columns.STATE_IDX].min()
        self.max_state = gt_trajectory[:, Columns.STATE_IDX].max()
        self.min_y = gt_trajectory.min()
        self.max_y = gt_trajectory.max()

        self.initialize_plot(gt_trajectory)

        self.num_resets += 1

    def post_reset(
        self,
        _: Tuple[torch.Tensor, Dict],
        _: Tuple[torch.Tensor, torch.Tensor],
    ):
        self.plot()

    def pre_close(self) -> str:
        self.save()
