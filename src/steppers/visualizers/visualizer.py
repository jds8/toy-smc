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
import time

from src.key_names.keys import Keys, Columns, VizKeys
from src.steppers.stepper import Stepper


class Visualizer(Stepper):
    def __init__(self, name: str, time_limit: int, **kwargs):
        self.name = name
        self.time_limit = time_limit
        self.time = 0

        self.min_state = None
        self.max_state = None
        self.min_y = None
        self.max_y = None
        self.fig = None
        self.gt_state = None
        self.gt_obs = None
        self.estimate = None
        self.estimate_gt = None
        self.estimate_lower = None
        self.estimate_upper = None
        self.variance_gt = None
        self.variance = None
        self.variance_lower = None
        self.variance_upper = None
        self.fig_display = None

        self.num_resets = 0

    def initialize_plot(
        self,
        gt_trajectory: torch.Tensor,
        estimator_stats: Tuple[torch.Tensor, torch.Tensor],
    ):
        clear_output(wait=True)

        estimate_was_none = self.estimate is None

        # Create initial figure
        self.fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        x_vals = np.arange(self.time_limit) + 1
        estimator_mean, estimator_std = estimator_stats
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
        self.estimate_gt = go.Scatter(
            x=x_vals,
            y=gt_trajectory[:, Columns.STATE_IDX].numpy(),
            mode='lines',
            name='Ground Truth Sequence',
            line=dict(color='blue')
        )
        self.estimate = go.Scatter(
            x=self.estimate.x if self.estimate is not None else [],
            y=self.estimate.y if self.estimate is not None else [],
            mode='lines',
            name='Filtering Estimate',
            line=dict(color='red')
        )
        self.estimate_lower = go.Scatter(
            x=self.estimate_lower.x if self.estimate_lower is not None else [],
            y=self.estimate_lower.y if self.estimate_lower is not None else [],
            mode='lines',
            name='Filtering Estimate',
            line=dict(dash='dash', width=0),
        )
        self.estimate_upper = go.Scatter(
            x=self.estimate_upper.x if self.estimate_upper is not None else [],
            y=self.estimate_upper.y if self.estimate_upper is not None else [],
            mode='lines',
            name='Filtering Estimate',
            line=dict(dash='dash', width=0),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.1)'
        )
        self.variance_gt = go.Scatter(
            x=x_vals,
            y=gt_trajectory[:, Columns.STATE_IDX].numpy(),
            mode='lines',
            name='Ground Truth Sequence',
            line=dict(color='blue')
        )
        self.variance = go.Scatter(
            x=x_vals if estimator_mean is not torch.nan else [],
            y=estimator_mean.squeeze().numpy() if estimator_mean is not torch.nan else [],
            mode='lines',
            name='Variance of Filtering Estimate',
            line=dict(color='red')
        )
        self.variance_lower = go.Scatter(
            x=x_vals if estimator_mean is not torch.nan else [],
            y=(estimator_mean - estimator_std).squeeze().numpy() if estimator_mean is not torch.nan else [],
            mode='lines',
            name='Variance of Filtering Estimate',
            line=dict(dash='dash', width=0),
        )
        self.variance_upper = go.Scatter(
            x=x_vals if estimator_mean is not torch.nan else [],
            y=(estimator_mean + estimator_std).squeeze().numpy() if estimator_mean is not torch.nan else [],
            mode='lines',
            name='Variance of Filtering Estimate',
            line=dict(dash='dash', width=0),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.1)'
        )
        self.fig.add_trace(self.gt_state, row=1, col=1)
        self.fig.add_trace(self.gt_obs, row=1, col=1)
        self.fig.add_trace(self.estimate, row=2, col=1)
        self.fig.add_trace(self.estimate_gt, row=2, col=1)
        self.fig.add_trace(self.estimate_lower, row=2, col=1)
        self.fig.add_trace(self.estimate_upper, row=2, col=1)
        self.fig.add_trace(self.variance, row=3, col=1)
        self.fig.add_trace(self.variance_gt, row=3, col=1)
        self.fig.add_trace(self.variance_lower, row=3, col=1)
        self.fig.add_trace(self.variance_upper, row=3, col=1)

        # set y-axis limits
        self.fig.update_xaxes(title_text='Time', range=[0, 500], row=3, col=1)

        # set y-axis limits
        self.fig.update_yaxes(range=[self.min_y, self.max_y], row=1, col=1)
        self.fig.update_yaxes(range=[self.min_y, self.max_y], row=2, col=1)
        self.fig.update_yaxes(range=[self.min_y, self.max_y], row=3, col=1)

        self.fig.update_traces()
        # update display
        self.fig_display = display(self.fig, display_id=True)

        if not estimate_was_none:
            self.estimate.x = []
            self.estimate.y = []
            self.estimate_lower.x = []
            self.estimate_lower.y = []
            self.estimate_upper.x = []
            self.estimate_upper.y = []

    def update(self, estimate_mean, estimate_std):
        self.estimate.x += (self.time,)
        self.estimate.y += (estimate_mean,)

        self.estimate_lower.x += (self.time,)
        self.estimate_lower.y += (estimate_mean - estimate_std,)

        self.estimate_upper.x += (self.time,)
        self.estimate_upper.y += (estimate_mean + estimate_std,)

        clear_output(wait=True)
        self.fig.update_layout(showlegend=False)

    def post_step(self, action, env_out, out: Dict):
        self.time += 1

        estimate_mean = out[VizKeys.STATE_MEAN]
        estimate_std = out[VizKeys.STATE_STD]

        self.update(estimate_mean, estimate_std)

    def save(self):
        if self.fig:
            img_name = '{}/{}_{}.pdf'.format(
                HydraConfig.get().run.dir,
                self.name,
                self.num_resets
            )
            self.fig.write_image(img_name, format='pdf')

    def post_reset(
        self,
        out: Tuple[torch.Tensor, Dict],
        estimator_stats: Tuple[torch.Tensor, torch.Tensor],
    ):
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

        self.initialize_plot(gt_trajectory, estimator_stats)

        self.num_resets += 1
        self.time = 0

    def pre_close(self) -> str:
        self.save()
