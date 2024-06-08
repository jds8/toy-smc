#!/usr/bin/env python3

from collections import namedtuple
from typing import List, Tuple, Optional
from dataclasses import dataclass

import torch


class Circle:
    def __init__(self, center: torch.Tensor, radius: torch.Tensor):
        self.center = center
        self.radius = radius

    def within_bounds(self, x: torch.Tensor):
        return torch.norm(self.center - x, dim=1, keepdim=True) <= self.radius

    def find_closest_point_on_circle(self, states, next_states):
        """
        define p(t) = states + t * (next_states - states)
        find that t_opt, using the quadratic formula, where
        p(t) lies on this circle and return p(t_opt)
        """

        diff = next_states - states
        B, D = diff[:, 0:1], diff[:, 1:]
        from_center = states - self.center
        A, C = from_center[:, 0:1], from_center[:, 1:]

        a = B**2 + D**2
        b = 2*(A*B + C*D)
        c = A**2 + B**2 - self.radius**2

        # solve quadratic
        pos_soln = (-b + torch.sqrt(b**2 - 4*a*c)) / 2*a
        neg_soln = (-b - torch.sqrt(b**2 - 4*a*c)) / 2*a

        # find those points between 0 and 1
        is_pos_between = torch.logical_and(0 < pos_soln, pos_soln < 1)

        # get those values where the "positive solution" is between 0 and 1
        # and take the negative solution where otherwise
        t = pos_soln.where(is_pos_between, neg_soln)

        next_states = states + t * diff
        return next_states


class Boundary:
    def __init__(self, lower: torch.Tensor, upper: torch.Tensor):
        self.lower = lower
        self.upper = upper

    def within_bounds(self, x: torch.Tensor):
        return torch.logical_and(self.lower <= x, x <= self.upper)


@dataclass
class Env:
    def __init__(
        self,
        num_particles: int,
        start_x: float,
        start_y: float,
        goal_x: float,
        goal_y: float,
        max_diff: float,
        time_limit: int,
        **kwargs,
    ):
        self.num_particles = num_particles
        self.start = torch.tensor([start_x, start_y]).expand(num_particles, 2)
        self.states = self.start
        self.goal = torch.tensor([goal_x, goal_y]).expand(num_particles, 2)
        self.max_diff = max_diff
        self.time_limit = torch.tensor([time_limit])
        self.time = torch.tensor([0])
        self.obstacle: Optional[Circle] = None
        self.x_boundary = Boundary(torch.tensor([0.]), torch.tensor([1.]))
        self.y_boundary = Boundary(torch.tensor([0.]), torch.tensor([1.]))
        self.done = torch.zeros(self.num_particles, 1, dtype=bool)

    def get_goal(self):
        """ There's only one goal, so just output the first element. """
        return self.goal[0, :]

    def place_obstacle(self, center: torch.Tensor, radius: torch.Tensor):
        self.obstacle = Circle(center, radius)

    def reward(self, s_next: torch.Tensor):
        reward = -((s_next - self.goal).norm(dim=1, keepdim=True) / (self.states - self.goal).norm(dim=1, keepdim=True)) ** 2
        return reward

    def set_state(self, states: torch.Tensor):
        self.states = states

    def transform_action(self, action: torch.Tensor):
        """
        Use a stereographic projection
        """
        denom = 1 + (action ** 2).sum(dim=1, keepdim=True)
        transformed_action = 2*action / denom
        scaled_action = transformed_action * self.max_diff
        return scaled_action

    def check_boundaries(self, next_states):
        next_states[:, 0:1] = torch.maximum(next_states[:, 0:1], self.x_boundary.lower)
        next_states[:, 0:1] = torch.minimum(next_states[:, 0:1], self.x_boundary.upper)
        next_states[:, 1:] = torch.maximum(next_states[:, 1:], self.y_boundary.lower)
        next_states[:, 1:] = torch.minimum(next_states[:, 1:], self.y_boundary.upper)

        # find points which intersect circle
        intersected_obstacle = self.obstacle.within_bounds(next_states)
        # get point on Circle closest to current point
        bounded_states = self.obstacle.find_closest_point_on_circle(self.states, next_states)
        # only update the points which intersected the circle
        next_states = bounded_states.where(
            torch.logical_and(
                intersected_obstacle,
                torch.logical_not(bounded_states.isnan())
            ),
            next_states
        )

        return next_states

    def step(self, action: torch.Tensor, update_state: bool):
        self.time += 1
        action = self.transform_action(action)
        assert (action.norm(dim=1, keepdim=True) <= self.max_diff + 1e-8).all()
        s_next = self.states + action
        s_next = self.check_boundaries(s_next)
        r = self.reward(s_next)
        done = (s_next - self.goal).norm(dim=1, keepdim=True) < self.max_diff
        if update_state:
            self.done = done
            self.states = s_next
        truncated = (self.time == self.time_limit).expand(done.shape)
        info = {}
        return s_next, r, done, truncated, info

    def set_resample_idx(self, idx: torch.Tensor):
        self.states = self.states[idx]

    def reset(self) -> Tuple[torch.Tensor, bool]:
        # resets the simulation and returns a new start state and done
        self.states = self.start
        self.done = torch.zeros(self.num_particles, 1, dtype=bool)
        self.time = torch.tensor([0])
        return self.states, 0., self.done, self.done, {}


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

    def step(self, action: torch.Tensor, update_state: bool):
        if update_state:
            self.trajectories['states'][-1].append(self.states)
            self.trajectories['actions'][-1].append(action)
            s_next, r, done, trunc, info = super().step(action, update_state)
            self.trajectories['rewards'][-1].append(r)
            self.trajectories['next_states'][-1].append(s_next)
            self.trajectories['dones'][-1].append(done)
        else:
            s_next, r, done, trunc, info = super().step(action, update_state)
        return s_next, r, done, trunc, info

    def set_resample_idx(self, idx: torch.Tensor):
        super().set_resample_idx(idx)
        self.trajectories['idx'][-1].append(idx)

    def reset(self):
        obs, r, done, truncated, info = super().reset()
        self.trajectories['states'].append([])
        self.trajectories['actions'].append([])
        self.trajectories['rewards'].append([])
        self.trajectories['next_states'].append([])
        self.trajectories['dones'].append([])
        self.trajectories['idx'].append([])
        return obs, r, done, truncated, info
