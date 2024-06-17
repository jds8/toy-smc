#!/usr/bin/env python3

import os

import stable_baselines3 as sb3

import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf, ListConfig


@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig):
    """
    Set the parameters and run the sim
    """
    log_level = cfg.get("log_level", "INFO")
    if log_level == "DEBUG":
        logging.basicConfig(level=logging.DEBUG)
    elif log_level == "INFO":
        logging.basicConfig(level=logging.INFO)
    elif log_level == "WARNING":
        logging.basicConfig(level=logging.WARNING)

    # Get logger *after* setting the level
    logger = logging.getLogger("main")
    # Print our config
    logger.info(f"CONFIG\n{OmegaConf.to_yaml(cfg)}")

    # Create the environment
    env = hydra.utils.instantiate(cfg.env)
    env.place_obstacle(torch.tensor([0.5, 0.5]), torch.tensor([0.20]))

    # Instantiate the agent
    model_name = 'models/ppo_env'
    if os.path.exists(model_name):
        model = sb3.load(model_name)
    else:
        model = sb3.PPO('MlpPolicy', env, verbose=1)

    # # Train the agent
    model.learn(total_timesteps=1000000)

    # # Save the agent
    model.save(model_name)


if __name__ == '__main__':
    main()

