#!/usr/bin/env python3

import pygame
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

    def point_to_window(self, point, scale):
        return tuple([x.item() * s for x, s in zip(point, scale)])

    def visualize(self, env: RecorderEnv):
        # Initialize Pygame
        pygame.init()

        # Set up display
        window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Particle Movement")

        for traj_idx, (traj, idxs) in enumerate(zip(
                env.trajectories['states'],
                env.trajectories['idx'],
        )):
            state = traj[0]
            for state_idx, (next_state, idx) in enumerate(zip(traj[1:], idxs[1:])):
                self.draw_window(window, env, next_state, traj_idx, state_idx)

        pygame.quit()

    def draw_window(self, window, env, particles, current_round, current_idx):
        # Clear the screen
        window.fill((255, 255, 255))

        # Draw the particles
        for obs in particles:
            pygame.draw.circle(
                window,
                self.BLUE,
                self.point_to_window(obs, scale=(self.width, self.height)),
                self.particle_radius)

        # Draw obstacles
        self.draw_obstacles(window, env)

        # Draw paths
        # self.draw_paths(window, env, current_round, current_idx)

        # Draw goal
        self.draw_goal(window, env)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        pygame.time.Clock().tick(60)

    def draw_goal(self, window, env):
        pygame.draw.circle(
            window,
            self.GREEN,
            self.point_to_window(env.get_goal(), scale=(self.width, self.height)),
            self.particle_radius
        )

    def draw_obstacles(self, window, env):
        for obstacle in env.obstacles:
            pygame.draw.circle(
                window,
                (0, 0, 0),
                self.point_to_window(obstacle.center, scale=(self.width, self.height)),
                self.point_to_window(obstacle.radius, scale=(self.height,))[0],
                width=self.obstacle_width
            )

    def draw_paths(self, window, env, current_round, current_idx):
        for traj_idx, (traj, next_traj) in enumerate(zip(
                env.trajectories['states'][:current_round+1],
                env.trajectories['next_states'][:current_round+1],
        )):
            color = self.BLUE if traj_idx == current_round else self.PINK
            for state, next_state in zip(traj, next_traj):
                for obs, next_obs in zip(state, next_state):
                    pygame.draw.circle(
                        window,
                        color,
                        self.point_to_window(obs, scale=(self.width, self.height)),
                        self.trail_width // 2,
                    )
                    # pygame.draw.line(
                    #     window,
                    #     color,
                    #     self.point_to_window(state, scale=(self.width, self.height)),
                    #     self.point_to_window(next_state, scale=(self.width, self.height)),
                    #     self.trail_width,
                    # )
