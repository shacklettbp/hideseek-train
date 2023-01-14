import madrona_python
import gpu_hideseek_python
import gym.spaces as spaces
import numpy as np
import torch
from habitat_baselines.common.tensor_dict import TensorDict


class MadronaVectorEnv:
    def __init__(self, config):
        self._config = config
        self._max_num_agents = 6
        self._num_envs = (
            config.habitat_baselines.num_environments // self._max_num_agents
        )
        if config.habitat_baselines.dry_run:
            self._sim = None
            self._action = torch.zeros((self._num_envs, 6, 5), device="cuda")
            self._reset = torch.zeros((self._num_envs, 3), device="cuda")
            self._reward = torch.zeros((self._num_envs, 6, 1), device="cuda")
            self._done = torch.zeros((self._num_envs, 1), device="cuda")

            # Observations
            self._box_data = torch.zeros(
                (self._num_envs, 6, 9, 7), device="cuda"
            )
            self._ramp_data = torch.zeros(
                (self._num_envs, 6, 2, 5), device="cuda"
            )
            self._agent_data = torch.zeros(
                (self._num_envs, 6, 5, 4), device="cuda"
            )
            self._prep_count = torch.zeros((self._num_envs, 1), device="cuda")

            # Masks
            self._valid_masks = torch.zeros(
                (self._num_envs, 6, 1), device="cuda"
            )
            self._agents_vis = torch.zeros(
                (self._num_envs, 6, 5, 1), device="cuda"
            )
            self._box_vis = torch.zeros(
                (self._num_envs, 6, 9, 1), device="cuda"
            )
            self._ramp_vis = torch.zeros(
                (self._num_envs, 6, 2, 1), device="cuda"
            )
        else:
            self._sim = gpu_hideseek_python.HideAndSeekSimulator(
                exec_mode=gpu_hideseek_python.ExecMode.CUDA,
                gpu_id=0,
                num_worlds=self._num_envs,
                min_entities_per_world=3,
                max_entities_per_world=3,
                render_width=64,
                render_height=64,
                debug_compile=False,
            )
            self._action = self._sim.action_tensor().to_torch()
            self._reset = self._sim.reset_tensor().to_torch()
            self._reward = self._sim.reward_tensor().to_torch()
            self._done = self._sim.done_tensor().to_torch()

            # Observations
            self._box_data = self._sim.box_data_tensor().to_torch()
            self._ramp_data = self._sim.ramp_data_tensor().to_torch()
            self._agent_data = self._sim.agent_data_tensor().to_torch()
            self._prep_count = self._sim.prep_counter_tensor().to_torch()

            # Masks
            self._valid_masks = self._sim.agent_mask_tensor().to_torch()
            self._agents_vis = (
                self._sim.visible_agents_mask_tensor().to_torch()
            )
            self._box_vis = self._sim.visible_boxes_mask_tensor().to_torch()
            self._ramp_vis = self._sim.visible_ramps_mask_tensor().to_torch()

        self._obs_shape = tuple(self._get_obs_no_cp().shape)[1:]

    @property
    def number_of_episodes(self):
        return self._config.habitat_baselines.test_episode_count

    @property
    def num_envs(self):
        return self._config.habitat_baselines.num_environments

    @property
    def action_spaces(self):
        return [spaces.MultiDiscrete([11, 11, 11, 2, 2])]

    @property
    def observation_spaces(self):
        return [
            spaces.Dict(
                {
                    "hideseek_state": spaces.Box(
                        shape=self._obs_shape,
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        dtype=np.float32,
                    )
                }
            )
        ]

    @property
    def orig_action_spaces(self):
        return self.action_spaces

    def _get_obs_no_cp(self):
        dat = [
            self._agents_vis * self._agent_data,
            self._box_vis * self._box_data,
            self._ramp_vis * self._ramp_data,
        ]
        dat = [x.view(self._num_envs, self._max_num_agents, -1) for x in dat]
        dat.append(
            self._prep_count.view(self._num_envs, 1, 1).repeat(
                1, self._max_num_agents, 1
            )
        )
        obs = torch.cat(dat, dim=-1)
        # Add agent to the batch dim
        return self._agent_batch(obs)

    def _agent_batch(self, X):
        return X.view(self._num_envs * self._max_num_agents, -1)

    def _get_obs(self):
        return TensorDict(hideseek_state=self._get_obs_no_cp().clone())

    def reset(self):
        # Reset all envs
        self._reset[:, 0] = 1
        self._reset[:, 1:2] = self._max_num_agents // 2
        self._internal_step()
        return self._get_obs()

    def _internal_step(self):
        if self._sim is not None:
            self._sim.step()

    def close(self):
        del self._sim

    def wait_step(self):
        return self._last

    def step(self, action):
        # Unbatch action.
        action = action.view(self._num_envs, self._max_num_agents, -1)
        self._action.copy_(action)

        # [0-10] -> [-5, 5] for the agent movement.
        for i in [0, 1, 2]:
            self._action[:, i] -= 5

        self._reset[:, :1] = self._done
        done = (
            self._done.clone()
            .view(-1, 1, 1)
            .repeat(1, self._max_num_agents, 1)
            .view(-1, 1)
            .cpu()
            .bool()
        )
        self._internal_step()
        obs = self._get_obs()
        reward = self._agent_batch(self._reward.clone()).cpu()
        info = [{} for _ in range(self._num_envs)]

        self._last = (obs, reward, done, info)
        return self._last


def construct_envs(
    config,
    workers_ignore_signals: bool = False,
    enforce_scenes_greater_eq_environments: bool = False,
):
    return MadronaVectorEnv(config)


def batch_obs(obs, device):
    return obs
