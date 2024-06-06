#!/usr/bin/env python3

import pygame
from src.policy import Policy
from src.env import RecorderEnv


class Visualizer:
    def __init__(self, num_rounds: int, env: RecorderEnv, policy: Policy, **kwargs):
        self.num_rounds = num_rounds
        self.env = env
        self.policy = policy
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

    def visualize(self):
        # Initialize Pygame
        pygame.init()

        # Set up display
        window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Particle Movement")

        for _ in range(self.num_rounds):
            obs, done = self.env.reset()
            self.draw_window(window, obs)

            while not done:
                action = self.policy.sample(obs, self.num_particles)
                obs, r, done, _, _ = self.env.step(action)

                self.draw_window(window, obs)

        pygame.quit()

    def draw_window(self, window, obs):
        x, y = obs[0].item(), obs[1].item()

        # Clear the screen
        window.fill((255, 255, 255))

        # Draw the particle
        pygame.draw.circle(
            window,
            self.BLUE,
            self.point_to_window(obs, scale=(self.width, self.height)),
            self.particle_radius)

        # Draw obstacles
        self.draw_obstacles(window)

        # Draw paths
        self.draw_paths(window)

        # Draw goal
        self.draw_goal(window)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        pygame.time.Clock().tick(60)

    def draw_goal(self, window):
        pygame.draw.circle(
            window,
            self.GREEN,
            self.point_to_window(self.env.goal, scale=(self.width, self.height)),
            self.particle_radius
        )

    def draw_obstacles(self, window):
        for obstacle in self.env.obstacles:
            pygame.draw.circle(
                window,
                (0, 0, 0),
                self.point_to_window(obstacle.center, scale=(self.width, self.height)),
                self.point_to_window(obstacle.radius, scale=(self.height,))[0],
                width=self.obstacle_width
            )

    def draw_paths(self, window):
        num_trajs = len(self.env.trajectories['states'])
        for traj_idx, (traj, next_traj) in enumerate(zip(
                self.env.trajectories['states'],
                self.env.trajectories['next_states'],
        )):
            color = self.BLUE if traj_idx == num_trajs - 1 else self.PINK
            for state, next_state in zip(traj, next_traj):
                pygame.draw.circle(
                    window,
                    color,
                    self.point_to_window(state, scale=(self.width, self.height)),
                    self.trail_width // 2,
                )
                pygame.draw.line(
                    window,
                    color,
                    self.point_to_window(state, scale=(self.width, self.height)),
                    self.point_to_window(next_state, scale=(self.width, self.height)),
                    self.trail_width,
                )
