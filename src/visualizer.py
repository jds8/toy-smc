#!/usr/bin/env python3

from abc import abstractmethod
from typing import Tuple

import pygame
from pygame.surface import Surface
import torch

from src.env import RecorderEnv


class Visualizer:
    def __init__(self, num_rounds: int, **kwargs):
        self.num_rounds = num_rounds
        self.width, self.height = 800, 600
        self.buffer = 10
        self.obstacle_width = 5
        self.particle_radius = 10
        self.trail_width = 3

        # colors
        self.BLUE = (0, 0, 255)
        self.PINK = (255, 0, 255)
        self.GREEN = (0, 255, 0)

    def point_to_window(self, point: torch.Tensor, scale: Tuple[float]):
        return tuple([x.item() * s for x, s in zip(point, scale)])

    def visualize(self, env: RecorderEnv):
        # Initialize Pygame
        pygame.init()

        # Set up display
        window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Particle Movement")

        for traj_idx, traj in enumerate(env.trajectories['states']):
            for state_idx, next_state in enumerate(traj):
                self.draw_window(window, env, next_state, traj_idx, state_idx)

        pygame.quit()

    def draw_window(
        self,
        window: Surface,
        env: RecorderEnv,
        particles: torch.Tensor,
        current_round: int,
        current_idx: int
    ):
        # Update display on resample
        self.draw_resample(window, env, current_round, current_idx)

        # Draw the particles
        for obs in particles:
            pygame.draw.circle(
                window,
                self.BLUE,
                self.point_to_window(obs, scale=(self.width, self.height)),
                self.particle_radius)

        # Draw obstacle
        self.draw_obstacle(window, env)

        # Draw paths
        self.draw_paths(window, env, current_round, current_idx)

        # Draw goal
        self.draw_goal(window, env)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        pygame.time.Clock().tick(40)

    def draw_resample(
        self,
        window: Surface, env: RecorderEnv,
        current_round: int,
        current_idx: int
    ):
        resample = env.trajectories['resampled'][current_round][current_idx]
        if resample:
            # indicate resample
            window.fill((200, 100, 0))
        else:
            # Clear the screen
            window.fill((255, 255, 255))

    def draw_goal(self, window: Surface, env: RecorderEnv):
        pygame.draw.circle(
            window,
            self.GREEN,
            self.point_to_window(env.get_goal(), scale=(self.width, self.height)),
            self.particle_radius
        )

    def draw_obstacle(self, window: Surface, env: RecorderEnv):
        obstacle = env.get_obstacle()
        top_left =  torch.tensor([
            self.width, self.height
        ]) * (obstacle.center - obstacle.radius)
        point = 2 * torch.concat([
            obstacle.radius, obstacle.radius
        ]) * torch.tensor([self.width, self.height])
        pygame.draw.ellipse(
            window,
            (0, 0, 0),
            pygame.Rect(
                top_left[0].item(),
                top_left[1].item(),
                point[0].item(),
                point[1].item()
            ),
            width=self.obstacle_width
        )

    @abstractmethod
    def draw_paths(
            self,
            window: Surface,
            env: RecorderEnv,
            current_round: int,
            current_idx: int
    ):
        pass


class CurrentPathsVisualizer(Visualizer):
    def draw_paths(
            self,
            window: Surface,
            env: RecorderEnv,
            current_round: int,
            current_idx: int
    ):
        traj = env.trajectories['states'][current_round]
        next_traj = env.trajectories['next_states'][current_round]
        idxs = env.trajectories['idx'][current_round]
        cur_idx = torch.arange(env.get_num_particles())
        for state, next_state, idx in zip(
                traj[current_idx-1::-1],
                next_traj[current_idx:0:-1],
                idxs[current_idx:0:-1]
        ):
            surviving_state = state[cur_idx]
            surviving_next_state = next_state[cur_idx]
            surviving_idx = torch.tensor(idx[cur_idx])
            for obs, next_obs, prev_idx in zip(
                    surviving_state,
                    surviving_next_state,
                    surviving_idx
            ):
                prev_state = state[prev_idx]
                pygame.draw.line(
                    window,
                    self.BLUE,
                    self.point_to_window(prev_state, scale=(self.width, self.height)),
                    self.point_to_window(next_obs, scale=(self.width, self.height)),
                    self.trail_width,
                )
                cur_idx = surviving_idx.unique()


class AllPathsVisualizer(Visualizer):
    def draw_paths(
            self,
            window: Surface,
            env: RecorderEnv,
            current_round: int,
            current_idx: int
    ):
        for traj_idx, (traj, next_traj, idxs) in enumerate(zip(
                env.trajectories['states'][:current_round+1],
                env.trajectories['next_states'][:current_round+1],
                env.trajectories['idx'][:current_round+1],
        )):
            color = self.BLUE if traj_idx == current_round else self.PINK
            state_idx = current_idx if traj_idx == current_round else len(traj)
            for state, next_state, idx in zip(
                    traj[state_idx-1::-1],
                    next_traj[state_idx:0:-1],
                    idxs[state_idx:0:-1]
            ):
                for obs, next_obs, prev_idx in zip(
                        state,
                        next_state,
                        idx
                ):
                    pygame.draw.circle(
                        window,
                        color,
                        self.point_to_window(obs, scale=(self.width, self.height)),
                        self.trail_width // 2,
                    )
                    prev_state = state[prev_idx]
                    pygame.draw.line(
                        window,
                        color,
                        self.point_to_window(prev_state, scale=(self.width, self.height)),
                        self.point_to_window(next_obs, scale=(self.width, self.height)),
                        self.trail_width,
                    )


class NoPathsVisualizer(Visualizer):
    def draw_paths(
            self,
            window: Surface,
            env: RecorderEnv,
            current_round: int,
            current_idx: int
    ):
        pass
