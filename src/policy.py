#!/usr/bin/env python3

import stable_baselines3 as sb3
import torch
import torch.distributions as dist


class Policy:
    def __init__(self, dim: int, **kwargs):
        self.dim = torch.tensor(dim)

    def sample(self, obs: torch.Tensor, num_particles: int) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(
            self,
            action: torch.Tensor,
            obs: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class PriorPolicy(Policy):
    def __init__(self, dim: int, **kwargs):
        super().__init__(dim)
        self.dist = dist.MultivariateNormal(
            torch.zeros(self.dim),
            torch.eye(self.dim)
        )

    def sample(self, obs: torch.Tensor, num_particles: int) -> torch.Tensor:
        action = self.dist.rsample([num_particles])
        log_prob = self.dist.log_prob(action)
        return action, log_prob

    def log_prob(self, action: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(action)


class SigmoidPolicy(PriorPolicy):
    def __init__(self, dim: int, max_value: float, **kwargs):
        super().__init__(dim)
        self.base_dist = self.dist
        self.max_value = max_value
        affine_transform = dist.AffineTransform(
            loc=-self.max_value / torch.sqrt(self.dim),
            scale=2 * self.max_value / torch.sqrt(self.dim),
        )
        transforms = [dist.SigmoidTransform(), affine_transform]
        self.dist = dist.TransformedDistribution(self.base_dist, transforms)


class TanhPolicy(PriorPolicy):
    def __init__(self, dim: int, max_value: float, **kwargs):
        super().__init__(dim)
        self.base_dist = self.dist
        self.max_value = max_value
        affine_transform = dist.AffineTransform(
            loc=0,
            scale=self.max_value / torch.sqrt(self.dim)
        )
        transforms = [dist.TanhTransform(), affine_transform]
        self.dist = dist.TransformedDistribution(self.base_dist, transforms)


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
            obs: torch.Tensor
    ) -> torch.Tensor:
        return self.model.policy.get_distribution(obs).log_prob(action)
