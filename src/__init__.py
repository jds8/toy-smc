#!/usr/bin/env python3

from . import envs, policies, sim, steppers, key_names

from gymnasium.envs.registration import register


register(
    id='ToyControlEnv-v0',
    entry_point='src.envs.toy_control_env.ToyControlEnv',
    max_episode_steps=100,
)
