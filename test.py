#!/usr/bin/env python3

import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf, ListConfig

from src.utils import register_resolvers


register_resolvers()


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

    # set up simulation
    sir_sim = hydra.utils.instantiate(cfg.sim)

    # run simulation
    output = sir_sim.run()
    logger.info(f"SIR output: {output}")

    # visualize simulation
    if 'visualizer' in cfg:
        visualizer = hydra.utils.instantiate(cfg.visualizer)
        if visualizer is not None:
            visualizer.visualize(sir_sim.env)


if __name__ == '__main__':
    main()
