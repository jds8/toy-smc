#!/usr/bin/env python3

from omegaconf import OmegaConf


def get_env(sim: str):
    if sim.endswith('.LinearGaussianSimulation'):
        return 'src.envs.env.StateSpaceEnv'
    elif sim.endswith('.SVSimulation'):
        return 'src.envs.stochastic_volatility_env.SVEnv'
    elif sim.endswith('.SIRSimulation'):
        return 'src.envs.toy_control_env.ToyControlEnv'
    else:
        raise NotImplementedError

def get_recorder(sim: str):
    if sim.endswith('.Simulation'):
        return 'src.steppers.recorders.recorder.Recorder'
    elif sim.endswith('SVSimulation'):
        return 'src.steppers.recorders.stochastic_volatility_recorder.SVRecorder'
    elif sim.endswith('SIRSimulation'):
        return 'src.recorders.toy_control_env.ToyControlRecorder'
    else:
        raise NotImplementedError

def get_visualizer(sim: str):
    if sim.endswith('SVSimulation'):
        return 'src.steppers.visualizers.stochastic_volatility_visualizer.SVVisualizer'
    elif sim.endswith('SIRSimulation'):
        return 'src.visualizer.toy_control_visualizer.ToyControlVisualizer'
    else:
        raise NotImplementedError

def get_prior(sim: str):
    if sim.endswith('.Simulation'):
        return 'src.policies.base_policy.GaussianPolicy'
    elif sim.endswith('SVSimulation'):
        return 'src.policies.stochastic_volatility_policies.SVPrior'
    elif sim.endswith('SIRSimulation'):
        return 'src.visualizer.toy_control_visualizer.ToyControlPrior'
    else:
        raise NotImplementedError


def register_resolvers():
    OmegaConf.register_new_resolver("get_env", get_env)
    OmegaConf.register_new_resolver("get_recorder", get_recorder)
    OmegaConf.register_new_resolver("get_visualizer", get_visualizer)
    OmegaConf.register_new_resolver("get_prior", get_prior)
