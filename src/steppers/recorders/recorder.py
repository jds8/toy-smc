#!/usr/bin/env python3

from typing import Tuple, Dict
import torch
import itertools

from hydra.core.hydra_config import HydraConfig

from src.steppers.stepper import Stepper
from src.key_names.keys import Keys, Columns, VizKeys


class Recorder:
    def __init__(self, name: str, num_particles: int, **kwargs):
        self.name = name
        self.num_particles = num_particles
        self.reset_trajectories()
        self.prev_log_weight = torch.tensor([0.])
        self.num_resets = 0
        self.sampled_trajectories = []

    def reset_trajectories(self):
        self.trajectories = {
            Keys.STATES: [],
            Keys.OBS: [],
            Keys.NEXT_STATES: [],
            Keys.LOG_LIK: [],
            Keys.LOG_PRIOR: [],
            Keys.LOG_PROPOSAL: [],
            Keys.IDX: [],
            Keys.RESAMPLED: [],
            Keys.DONE: [],
            Keys.GT_TRAJECTORY: [],
        }

    def post_step(self, _: torch.Tensor, env_out: Tuple):
        _, _, _, _, info = env_out
        self.trajectories[Keys.STATES].append(info[Keys.STATES])
        self.trajectories[Keys.OBS].append(info[Keys.OBS])
        self.trajectories[Keys.NEXT_STATES].append(info[Keys.NEXT_STATES])
        self.trajectories[Keys.LOG_LIK].append(info[Keys.LOG_LIK])
        self.trajectories[Keys.LOG_PRIOR].append(info[Keys.LOG_PRIOR])
        self.trajectories[Keys.LOG_PROPOSAL].append(info[Keys.LOG_PROPOSAL])
        self.trajectories[Keys.IDX].append(info[Keys.IDX])
        self.trajectories[Keys.RESAMPLED].append(info[Keys.RESAMPLED])
        self.trajectories[Keys.DONE].append(info[Keys.DONE])
        self.trajectories[Keys.GT_TRAJECTORY].append(info[Keys.GT_TRAJECTORY])

        gt_trajectory = info[Keys.GT_TRAJECTORY]

        log_weight = info[Keys.LOG_LIK] + info[Keys.LOG_PRIOR] - info[Keys.LOG_PROPOSAL]
        N = torch.tensor(log_weight.shape[0])
        log_evidence = self.prev_log_weight + log_weight.sum() - N.log()
        weight = log_weight.exp()
        state_mean = (weight * info[Keys.NEXT_STATES]).mean()
        state_std = (weight * (info[Keys.NEXT_STATES] - state_mean) ** 2).sum() / (N-1)
        self.prev_log_weight = torch.tensor([0.]) if info[Keys.RESAMPLED][0] else log_evidence

        return {
            VizKeys.STATE_MEAN: state_mean,
            VizKeys.STATE_STD: state_std,
            VizKeys.LOG_EVIDENCE: log_evidence,
            Keys.GT_TRAJECTORY: gt_trajectory,
        }

    def compute_estimator_stats(self):
        mean, std = torch.nan, torch.nan
        if self.sampled_trajectories:
            stacked_trajs = torch.hstack(self.sampled_trajectories)
            mean = stacked_trajs.mean(dim=1, keepdim=True)
            std = stacked_trajs.std(dim=1, keepdim=True)
        return mean, std

    def post_reset(self, _: Tuple) -> str:
        if self.trajectories[Keys.STATES]:
            # convert
            states = torch.stack(self.trajectories[Keys.STATES])
            obs = torch.stack(self.trajectories[Keys.OBS])
            next_states = torch.stack(self.trajectories[Keys.NEXT_STATES])
            log_lik = torch.stack(self.trajectories[Keys.LOG_LIK])
            log_prior = torch.stack(self.trajectories[Keys.LOG_PRIOR])
            log_proposal = torch.stack(self.trajectories[Keys.LOG_PROPOSAL])
            idx = torch.stack(self.trajectories[Keys.IDX])
            resampled = torch.stack(self.trajectories[Keys.RESAMPLED])
            done = torch.stack(self.trajectories[Keys.DONE])
            gt_trajectory = self.trajectories[Keys.GT_TRAJECTORY][-1]

            self.sampled_trajectories.append(next_states)

            list_of_infos = [{
                "{}_{}".format(Keys.STATES, particle_idx): states[:, particle_idx, :],
                "{}_{}".format(Keys.OBS, particle_idx): obs[:, particle_idx, :],
                "{}_{}".format(Keys.NEXT_STATES, particle_idx): next_states[:, particle_idx, :],
                "{}_{}".format(Keys.LOG_LIK, particle_idx): log_lik[:, particle_idx, :],
                "{}_{}".format(Keys.LOG_PRIOR, particle_idx): log_prior[:, particle_idx, :],
                "{}_{}".format(Keys.LOG_PROPOSAL, particle_idx): log_proposal[:, particle_idx, :],
                "{}_{}".format(Keys.IDX, particle_idx): idx[:, particle_idx, :],
                "{}_{}".format(Keys.RESAMPLED, particle_idx): resampled[:, particle_idx, :],
                "{}_{}".format(Keys.DONE, particle_idx): done[:, particle_idx, :],
            } for particle_idx in range(self.num_particles)]

            list_of_infos.append({
                "{}_{}".format(
                    Keys.GT_TRAJECTORY,
                    Columns.STATE_IDX
                ): gt_trajectory[:, Columns.STATE_IDX],
                "{}_{}".format(
                    Keys.GT_TRAJECTORY,
                    Columns.OBS_IDX
                ): gt_trajectory[:, Columns.OBS_IDX],
            })

            combined_info = dict(
                itertools.chain.from_iterable(
                    d.items() for d in list_of_infos
                )
            )

            data_path_name = '{}/{}_{}.pt'.format(
                HydraConfig.get().run.dir,
                self.name,
                self.num_resets
            )
            torch.save(combined_info, data_path_name)

        self.num_resets += 1
        self.reset_trajectories()

        estimator_stats = self.compute_estimator_stats()

        return estimator_stats

    def pre_close(self) -> str:
        return self.post_reset(None)
