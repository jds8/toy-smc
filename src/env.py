#!/usr/bin/env python3

from collections import namedtuple
from typing import List, Tuple, Optional
from dataclasses import dataclass

import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.policy import Policy


class Circle:
    def __init__(self, center: torch.Tensor, radius: torch.Tensor):
        self.center = center
        self.radius = radius

    def get_center(self):
        return self.center

    def get_radius(self):
        return self.radius

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

    def get_lower(self):
        return self.lower

    def get_upper(self):
        return self.upper

    def within_bounds(self, x: torch.Tensor):
        return torch.logical_and(self.lower <= x, x <= self.upper)


@dataclass
class Env(gym.Env):
    def __init__(
        self,
        prior_policy: Policy,
        num_particles: int,
        start_x: float,
        start_y: float,
        goal_x: float,
        goal_y: float,
        max_diff: float,
        time_limit: int,
        **kwargs,
    ):
        super().__init__()

        self.action_space = spaces.Box(
            low=-max_diff,
            high=max_diff,
            shape=(num_particles, 2),
            dtype=float
        )
        self.observation_space = spaces.Box(
            low=0.,
            high=1.,
            shape=(num_particles, 2),
            dtype=np.float32
        )

        self.prior_policy = prior_policy
        self.num_particles = num_particles
        self.start = torch.tensor([start_x, start_y]).expand(num_particles, 2)
        self.states = self.start
        self.goal = torch.tensor([goal_x, goal_y]).expand(num_particles, 2)
        self.max_diff = max_diff
        self.time_limit = torch.tensor([time_limit], dtype=int)
        self.time = torch.tensor([0])
        self.obstacle: Optional[Circle] = None
        self.x_boundary = Boundary(torch.tensor([0.]), torch.tensor([1.]))
        self.y_boundary = Boundary(torch.tensor([0.]), torch.tensor([1.]))
        self.done = torch.zeros(self.num_particles, 1, dtype=bool)

        self.seed = kwargs.seed if 'seed' in kwargs else 100
        torch.manual_seed(self.seed)

    def get_goal(self):
        """ There's only one goal, so just output the first element. """
        return self.goal[0, :]

    def place_obstacle(self, center: torch.Tensor, radius: torch.Tensor):
        self.obstacle = Circle(center, radius)

    def reward(self, s_next: torch.Tensor):
        reward = -((s_next - self.goal).norm(dim=1) / (self.states - self.goal).norm(dim=1)) ** 2
        return reward

    def set_state(self, states: torch.Tensor):
        self.states = states

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

    def step(self, action: np.ndarray, update_state: bool = True):
        assert self.action_space.contains(action), "Invalid action"
        self.time += 1
        log_prior = self.prior_policy.log_prob(
            torch.tensor(action),
            self.states
        )
        s_next = self.states + action
        s_next = self.check_boundaries(s_next)
        log_lik = self.reward(s_next)
        log_joint = log_lik + log_prior
        done = (s_next - self.goal).norm(dim=1, keepdim=True) < self.max_diff
        truncated = (self.time == self.time_limit).expand(done.shape)
        if update_state:
            self.done = done
            self.states = s_next
        info = {'states': s_next, 'log_lik': log_lik, 'log_prior': log_prior, 'done': done}
        return s_next, log_joint, done, truncated, info

    def resample_step(
        self,
        action: np.ndarray,
        resample_idx: torch.Tensor,
    ):
        self.states = self.states[resample_idx]
        action = action[resample_idx]
        return self.step(action, update_state=True)

    def reset(self, seed: int | None = None) -> Tuple[torch.Tensor, bool]:
        # resets the simulation and returns a new start state and done
        self.states = self.start
        self.done = torch.zeros(self.num_particles, 1, dtype=bool)
        self.time = torch.tensor([0])
        info = {'states': self.start, 'r': self.done.to(float), 'done': self.done}
        return self.states, info

    def set_from_info(self, info: dict):
        self.states = info['states']
        self.done = info['done']

    def get_num_particles(self):
        return self.num_particles

    def get_obstacle(self):
        return self.obstacle

    def get_time_limit(self):
        return self.time_limit


class RecorderEnv(Env):
    STATES = 'states'
    ACTIONS = 'actions'
    REWARDS = 'rewards'
    NEXT_STATES = 'next_states'
    DONES = 'dones'
    IDX = 'idx'
    RESAMPLED = 'resampled'
    WEIGHTS = 'weights'

    def __init__(self, inner_env: Env, **kwargs):
        self.env = inner_env
        self.trajectories = {
            self.STATES: [],
            self.ACTIONS: [],
            self.REWARDS: [],
            self.NEXT_STATES: [],
            self.DONES: [],
            self.IDX: [],
            self.RESAMPLED: [],
            self.WEIGHTS: [],
        }
        self.temp_data = {
            self.STATES: None,
            self.ACTIONS: None,
            self.REWARDS: None,
            self.NEXT_STATES: None,
            self.DONES: None,
            self.IDX: None,
            self.RESAMPLED: None,
            self.WEIGHTS: None,
        }

    def place_obstacle(self, center: torch.Tensor, radius: torch.Tensor):
        self.env.place_obstacle(center, radius)

    def step(self, action: torch.Tensor, update_state: bool = True):
        if update_state:
            self.trajectories[self.STATES][-1].append(self.env.states)
            self.trajectories[self.ACTIONS][-1].append(action)
            s_next, r, done, trunc, info = self.env.step(action, update_state)
            self.trajectories[self.REWARDS][-1].append(r)
            self.trajectories[self.NEXT_STATES][-1].append(s_next)
            self.trajectories[self.DONES][-1].append(done)
            self.trajectories[self.IDX][-1].append(torch.arange(self.env.states.shape[0]))
            self.trajectories[self.RESAMPLED][-1].append(False)
        else:
            self.temp_data[self.STATES] = self.env.states
            self.temp_data[self.ACTIONS] = action
            s_next, r, done, trunc, info = self.env.step(action, update_state)
            self.temp_data[self.REWARDS] = r
            self.temp_data[self.NEXT_STATES] = s_next
            self.temp_data[self.DONES] = done
            self.temp_data[self.IDX] = torch.arange(self.env.states.shape[0])
            self.temp_data[self.RESAMPLED] = False
        return s_next, r, done, trunc, info

    def resample_step(
        self,
        action: torch.Tensor,
        resample_idx: torch.Tensor,
    ):
        self.trajectories[self.STATES][-1].append(self.env.states[resample_idx])
        self.trajectories[self.ACTIONS][-1].append(action[resample_idx])
        obs, r, done, trunc, info = self.env.resample_step(action, resample_idx)
        self.trajectories[self.REWARDS][-1].append(r)
        self.trajectories[self.NEXT_STATES][-1].append(obs)
        self.trajectories[self.DONES][-1].append(done)
        self.trajectories[self.IDX][-1].append(resample_idx)
        self.trajectories[self.RESAMPLED][-1].append(True)
        return obs, r, done, trunc, info

    def reset(self):
        obs = self.env.reset()
        self.trajectories[self.STATES].append([])
        self.trajectories[self.ACTIONS].append([])
        self.trajectories[self.REWARDS].append([])
        self.trajectories[self.NEXT_STATES].append([])
        self.trajectories[self.DONES].append([])
        self.trajectories[self.IDX].append([])
        self.trajectories[self.RESAMPLED].append([])
        self.trajectories[self.WEIGHTS].append([])
        return obs

    def get_num_particles(self):
        return self.env.get_num_particles()

    def get_obstacle(self):
        return self.env.get_obstacle()

    def get_time_limit(self):
        return self.env.get_time_limit()

    def get_goal(self):
        return self.env.get_goal()

    def set_from_info(self, info: dict):
        self.env.set_from_info(info)
        self.trajectories[self.STATES][-1].append(self.temp_data[self.STATES])
        self.trajectories[self.ACTIONS][-1].append(self.temp_data[self.ACTIONS])
        self.trajectories[self.REWARDS][-1].append(self.temp_data[self.REWARDS])
        self.trajectories[self.NEXT_STATES][-1].append(self.temp_data[self.NEXT_STATES])
        self.trajectories[self.DONES][-1].append(self.temp_data[self.DONES])
        self.trajectories[self.IDX][-1].append(self.temp_data[self.IDX])
        self.trajectories[self.RESAMPLED][-1].append(self.temp_data[self.RESAMPLED])

    def store_weights(self, weights: torch.Tensor):
        self.trajectories[self.WEIGHTS][-1].append(weights)
