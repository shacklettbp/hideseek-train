import madrona_python
import gpu_hideseek_python
import gym.spaces as spaces
import numpy as np
import torch
from habitat_baselines.common.tensor_dict import TensorDict
from collections import defaultdict


class MadronaVectorEnv:
    def __init__(self, config):
        self._config = config
        self._debug_env = self._config.habitat_baselines.debug_env
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
            self._agent_type = torch.zeros(
                (self._num_envs, 6, 1), device="cuda"
            )

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
            self._rgb = torch.zeros(
                (self._num_envs, 6, 64, 64, 4), device="cuda"
            )

        else:
            print(f"Allocating {self._num_envs} environments")
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
            self._rgb = self._sim.rgb_tensor().to_torch()
            self._reset = self._sim.reset_tensor().to_torch()
            self._reward = self._sim.reward_tensor().to_torch()
            self._done = self._sim.done_tensor().to_torch()
            self._agent_type = self._sim.agent_type_tensor().to_torch()

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

        self._start_ramp_pos = None
        self._start_box_pos = None
        self._start_agent_pos = None
        self._actions_debug = defaultdict(list)

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

    def _get_ramp_pos(self):
        return self._ramp_data[:, 0, :, :2]

    def _get_box_pos(self):
        return self._box_data[:, 0, :, :2]

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
        dat.extend(
            [
                self._prep_count.view(self._num_envs, 1, 1).repeat(
                    1, self._max_num_agents, 1
                ),
                # self._agent_type,
            ]
        )
        obs = torch.cat(dat, dim=-1)
        # Add agent to the batch dim
        return self._agent_batch(obs)

    def _agent_batch(self, X):
        return X.view(self._num_envs * self._max_num_agents, -1)

    def _get_obs(self):
        obs = self._get_obs_no_cp()
        if self._debug_env and torch.isnan(obs).any():
            # Observation returned nan
            print(
                "Bad envs",
                [
                    i
                    for i in range(self._agent_data.shape[0])
                    if torch.isnan(self._agent_data[i]).any()
                ],
            )
            # Access action sequence from `self._actions_debug`.
            breakpoint()
        obs = torch.nan_to_num(obs)

        return TensorDict(hideseek_state=obs.clone())

    def reset(self):
        # Reset all envs
        self._reset[:, 0] = 1
        # Set the number of hiders
        self._reset[:, 1] = 3
        # Set the number of seekers
        self._reset[:, 2] = 3
        self._internal_step()
        self._num_steps = torch.zeros((self._num_envs, 1), device="cuda")

        self._start_box_pos = self._get_box_pos()
        self._start_ramp_pos = self._get_ramp_pos()
        return self._get_obs()

    def _update_start_pos(self):
        is_reset = self._reset[:, 0].view(-1, 1, 1)
        self._start_box_pos = ((1.0 - is_reset) * self._start_box_pos) + (
            is_reset * self._get_box_pos()
        )
        self._start_ramp_pos = ((1.0 - is_reset) * self._start_ramp_pos) + (
            is_reset * self._get_ramp_pos()
        )

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

        # [0-10] -> [-5, 5] for the agent movement (translation and roation).
        for i in [0, 1, 2]:
            self._action[:, i] -= 5

        self._reset[:, :1] = self._done
        self._num_steps += 1.0
        self._num_steps *= 1.0 - self._done
        done = (
            self._done.clone()
            .view(-1, 1, 1)
            .repeat(1, self._max_num_agents, 1)
            .view(-1, 1)
            .cpu()
            .bool()
        )
        self._update_start_pos()

        if self._debug_env:
            # Save action sequences if in debug mode.
            for env_i in range(self._num_envs):
                self._actions_debug[env_i].append(self._action[env_i].cpu())
            if done[env_i].item():
                self._actions_debug[env_i] = []

        self._internal_step()
        obs = self._get_obs()
        reward = self._reward.clone()

        ramp_dist = torch.linalg.norm(
            self._get_ramp_pos() - self._start_ramp_pos, dim=-1
        ).mean(-1)
        box_dist = torch.linalg.norm(
            self._get_box_pos() - self._start_box_pos, dim=-1
        ).mean(-1)

        info = [
            {
                "r_t": reward[env_i, agent_i].item(),
                "ns": self._num_steps[env_i].item(),
                "stage": self._prep_count[env_i].item(),
                # "ramp_dist": ramp_dist[env_i].item(),
                # "box_dist": box_dist[env_i].item(),
            }
            for env_i in range(self._num_envs)
            for agent_i in range(self._max_num_agents)
        ]

        reward = self._agent_batch(reward).cpu()

        self._last = (obs, reward, done, info)
        return self._last

    def render(self):
        return self._rgb


def construct_envs(
    config,
    workers_ignore_signals: bool = False,
    enforce_scenes_greater_eq_environments: bool = False,
):
    return MadronaVectorEnv(config)


def batch_obs(obs, device):
    return obs
