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


class BaseVisualizer(Stepper):
    def __init__(
        self,
        name: str,
        data_path: str,
        title: str,
        **kwargs
    ):
        self.name = name
        self.title = title

        self.time_limit = 0
        self.num_resets = 0

        self.fig = None

        self.ground_truth = None
        self.mean = None
        self.upper_ci = None
        self.lower_ci = None

        self.gt_data = None
        self.mean_data = None
        self.upper_ci_data = None
        self.lower_ci_data = None
        self.times = None

    def load_data(self, data_path: str):
        """
        Sets:
        gt_data, mean_data, upper_ci_data, lower_ci_data, time_limit
        """
        raise NotImplementedError

    def make_plot(self):
        # Create initial figure
        self.fig = make_subplots(rows=1, cols=1)
        self.ground_truth = go.Scatter(
            x=self.times,
            y=self.gt_data,
            mode='lines',
            name='Ground Truth',
            line=dict(color='blue')
        )
        self.mean = go.Scatter(
            x=self.times,
            y=self.mean_data,
            mode='lines',
            name=self.title,
            line=dict(color='red')
        )
        # self.upper_ci = go.Scatter(
        #     x=self.times,
        #     y=self.upper_ci_data,
        #     mode='lines',
        #     line=dict(dash='dash', width=0),
        #     showlegend=False,
        # )
        # self.lower_ci = go.Scatter(
        #     x=self.times,
        #     y=self.lower_ci_data,
        #     mode='lines',
        #     line=dict(dash='dash', width=0),
        #     name='One Standard Deviation',
        #     fill='tonexty',
        #     fillcolor='rgba(255, 0, 0, 0.1)'
        # )
        self.fig.add_trace(self.ground_truth, row=1, col=1)
        self.fig.add_trace(self.mean, row=1, col=1)
        # self.fig.add_trace(self.upper_ci, row=1, col=1)
        # self.fig.add_trace(self.lower_ci, row=1, col=1)

        self.fig.update_traces()
        display(self.fig, display_id=True)

        self.save()

    def save(self):
        if self.fig:
            img_name = '{}/{}_{}.pdf'.format(
                HydraConfig.get().run.dir,
                self.name,
                self.num_resets
            )
            self.fig.write_image(img_name, format='pdf')
