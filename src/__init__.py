#!/usr/bin/env python3

from . import env, policy, visualizer, sim

from gymnasium.envs.registration import register

register(
    id='Env-v0',
    entry_point='src.env.Env',
    max_episode_steps=100,
)
