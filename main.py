#!/usr/bin/env python3

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

    # set up simulation
    sim = hydra.utils.instantiate(cfg.sim)
    sim.env.place_obstacle(torch.tensor([0.5, 0.5]), torch.tensor([0.20]))

    # run simulation
    sim.run()

    # visualize simulation
    visualizer = hydra.utils.instantiate(cfg.visualizer)
    visualizer.visualize(sim.env)


if __name__ == '__main__':
    main()
