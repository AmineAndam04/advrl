import numpy as np
from copy import deepcopy
from gymnasium.vector.utils import batch_space


class VecEnv:
    def __init__(self, env_fns):
        """Similar to AutoresetMode.SAME_STEP"""
        self.env_fns = env_fns
        self.envs = [env_fn() for env_fn in env_fns]
        self.num_envs = len(self.envs)
        self.single_action_space = self.envs[0].action_space
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.single_observation_space = self.envs[0].observation_space

    def reset_ith_env(self, env_idx, seed):
        obs, infos = self.envs[env_idx].reset(seed=seed)
        return obs, infos

    def reset(self, seed=None):
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        elif isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        obs = [None for _ in range(self.num_envs)]
        infos = [None for _ in range(self.num_envs)]
        for i, (env, env_seed) in enumerate(zip(self.envs, seed)):
            obs[i], infos[i] = env.reset(seed=env_seed)
        return np.stack(obs), infos

    def step(self, actions):
        observation = [None for _ in range(self.num_envs)]
        reward = [None for _ in range(self.num_envs)]
        terminated = [None for _ in range(self.num_envs)]
        truncated = [None for _ in range(self.num_envs)]
        info = [None for _ in range(self.num_envs)]
        for i, env in enumerate(self.envs):
            observation[i], reward[i], terminated[i], truncated[i], info[i] = env.step(actions[i])
            if terminated[i] or truncated[i]:
                temp_obs, _ = env.reset()
                info[i]["terminal_obs"] = deepcopy(observation[i])
                observation[i] = temp_obs
        return (
            np.stack(observation),
            np.stack(reward),
            np.stack(terminated),
            np.stack(truncated),
            info,
        )

    def call(self, name, *args, **kwargs):
        results = []
        for env in self.envs:
            function = env.get_wrapper_attr(name)

            if callable(function):
                results.append(function(*args, **kwargs))
            else:
                results.append(function)
        return tuple(results)

    def get_attr(self, name):
        return self.call(name)

    def set_attr(self, name, values):
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]

        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the number of environments. "
                f"Got `{len(values)}` values for {self.num_envs} environments."
            )

        for env, value in zip(self.envs, values):
            env.set_wrapper_attr(name, value)

    def close_extras(self, **kwargs):
        """Close the environments."""
        if hasattr(self, "envs"):
            [env.close() for env in self.envs]


class EpisodicVecEnv:
    def __init__(self, env_fns):
        self.env_fns = env_fns
        self.envs = [env_fn() for env_fn in env_fns]
        self.num_envs = len(self.envs)
        self.single_action_space = self.envs[0].action_space
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.single_observation_space = self.envs[0].observation_space

    def reset(self, seed=None):
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        elif isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        obs = [None for _ in range(self.num_envs)]
        infos = [None for _ in range(self.num_envs)]
        for i, (env, env_seed) in enumerate(zip(self.envs, seed)):
            obs[i], infos[i] = env.reset(seed=env_seed)
        self.alive_envs = list(range(self.num_envs))
        return np.stack(obs), infos

    def step(self, actions):
        """
        S_0,A_0,R_1,S_1,A_1,R_2 ......S_{T-1},A_{T-1},R_T,S_T
        """
        observation = [None for _ in range(len(self.alive_envs))]
        reward = [None for _ in range(len(self.alive_envs))]
        terminated = [None for _ in range(len(self.alive_envs))]
        truncated = [None for _ in range(len(self.alive_envs))]
        info = [None for _ in range(len(self.alive_envs))]

        for i, j in enumerate(self.alive_envs[:]):
            observation[i], reward[i], terminated[i], truncated[i], info[i] = self.envs[j].step(
                actions[i]
            )
            if terminated[i] or truncated[i]:
                self.alive_envs.remove(j)
        return (
            np.stack(observation),
            np.stack(reward),
            np.stack(terminated),
            np.stack(truncated),
            info,
        )

    def call(self, name, *args, **kwargs):
        results = []
        for env in self.envs:
            function = env.get_wrapper_attr(name)

            if callable(function):
                results.append(function(*args, **kwargs))
            else:
                results.append(function)
        return tuple(results)

    def get_attr(self, name):
        return self.call(name)

    def set_attr(self, name, values):
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]

        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the number of environments. "
                f"Got `{len(values)}` values for {self.num_envs} environments."
            )

        for env, value in zip(self.envs, values):
            env.set_wrapper_attr(name, value)

    def close_extras(self, **kwargs):
        """Close the environments."""
        if hasattr(self, "envs"):
            [env.close() for env in self.envs]
