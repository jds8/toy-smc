#!/usr/bin/env python3

from collections import namedtuple
from typing import List, Tuple, Optional

import torch


class Obstacle:
    def __init__(self, center: torch.Tensor, radius: torch.Tensor):
        self.center = center
        self.radius = radius

    def within_bounds(self, x: torch.Tensor):
        return torch.norm(self.center - x, dim=1) < self.radius


class Boundary:
    def __init__(self, lower: torch.Tensor, upper: torch.Tensor):
        self.lower = lower
        self.upper = upper

    def within_bounds(self, x: torch.Tensor):
        return torch.logical_and(self.lower <= x, x <= self.upper)


class Env:
    def __init__(
        self,
        num_particles: int,
        start_x: float,
        start_y: float,
        goal_x: float,
        goal_y: float,
        max_diff: float,
        **kwargs,
    ):
        self.num_particles = num_particles
        self.start = torch.tensor([start_x, start_y]).expand(num_particles, 2)
        self.states = self.start
        self.goal = torch.tensor([goal_x, goal_y]).expand(num_particles, 2)
        self.max_diff = max_diff
        self.obstacles: List[Obstacle] = []
        self.x_boundary = Boundary(0., 1.)
        self.y_boundary = Boundary(0., 1.)
        self.done = torch.zeros_like(self.states, dtype=bool)

    def place_obstacle(self, center: torch.Tensor, radius: torch.Tensor):
        o = Obstacle(center, radius)
        self.obstacles.append(o)

    def reward(self, s_next: torch.Tensor):
        return -((s_next - self.goal).norm(dim=1, keepdim=True) / (self.states - self.goal).norm(dim=1, keepdim=True)) ** 2

    def set_state(self, states):
        self.states = states

    def transform_action(self, action):
        """
        Use a stereographic projection
        """
        denom = 1 + torch.sqrt(1 + (action ** 2).sum(dim=1, keepdim=True))
        transformed_action = action / denom
        scaled_action = transformed_action * self.max_diff
        return scaled_action

    def step(self, action, update_state):
        action = self.transform_action(action)
        assert action.norm(dim=1, keepdim=True) <= self.max_diff
        s_next = self.states + action
        r = self.reward(s_next)
        if update_state:
            self.states = s_next
        at_goal = (self.states - self.goal).norm(dim=1, keepdim=True) < self.max_diff
        within_x = self.x_boundary.within_bounds(self.states[0])
        within_y = self.y_boundary.within_bounds(self.states[1])
        within_bounds = torch.logical_and(within_x, within_y)
        intersected_obstacle = torch.hstack([obstacle.within_bounds(self.states) for obstacle in self.obstacles]).any(dim=1, keepdim=True)
        done = torch.logical_or(torch.logical_or(at_goal, not within_bounds), intersected_obstacle)
        done = torch.logical_or(self.done, done)
        if update_state:
            self.done = done
        truncated = torch.zeros_like(done, dtype=bool)
        info = {}
        return s_next, r, done, truncated, info

    def set_resample_idx(self, idx: torch.Tensor):
        self.states = self.states[idx]

    def reset(self) -> Tuple[torch.Tensor, bool]:
        # resets the simulation and returns a new start state and done
        self.states = self.start
        return self.states, torch.zeros_like(self.states, dtype=bool)


class RecorderEnv(Env):
    def __init__(
        self,
        num_particles: int,
        start_x: float,
        start_y: float,
        goal_x: float,
        goal_y: float,
        max_diff: float,
        **kwargs,
    ):
        super().__init__(num_particles, start_x, start_y, goal_x, goal_y, max_diff, **kwargs)
        self.trajectories = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'idx': []
        }

    def step(self, action, update_state):
        if update_state:
            self.trajectories['states'][-1].append(self.states)
            self.trajectories['actions'][-1].append(action)
            s_next, r, done, trunc, info = super().step(action, update_state)
            self.trajectories['rewards'][-1].append(r)
            self.trajectories['next_states'][-1].append(s_next)
            self.trajectories['dones'][-1].append(done)
            return s_next, r, done, trunc, info

    def set_resample_idx(self, idx: torch.Tensor):
        super().set_resample_idx(idx)
        self.trajectories['idx'][-1].append(idx)

    def reset(self):
        obs, done = super().reset()
        self.trajectories['states'].append([])
        self.trajectories['actions'].append([])
        self.trajectories['rewards'].append([])
        self.trajectories['next_states'].append([])
        self.trajectories['dones'].append([])
        self.trajectories['idx'].append([])
        return obs, done
