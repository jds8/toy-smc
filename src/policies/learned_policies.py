#!/usr/bin/env python3

import stable_baselines3 as sb3
import torch

from src.policies.base_policy import Policy


class RLPolicy(Policy):
    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.model = sb3.PPO.load(model_path)
        dim = self.model.action_space.shape[1]
        super().__init__(dim)

    def sample(self, obs: torch.Tensor, num_particles: int) -> torch.Tensor:
        actions, _ = self.model.predict(obs)
        tensor_action = torch.tensor(actions)
        device = self.model.policy.device
        log_prob = self.log_prob(tensor_action.to(device), obs.to(device))
        return tensor_action, log_prob.to(obs.device)

    def log_prob(
            self,
            action: torch.Tensor,
            t: int,
            obs: torch.Tensor
    ) -> torch.Tensor:
        return self.model.policy.get_distribution(obs).log_prob(action)
