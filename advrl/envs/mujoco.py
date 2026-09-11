import gymnasium as gym
import numpy as np
from gymnasium.wrappers.utils import RunningMeanStd


def create_make_env_mujoco_from_json(
    wrappers, envparams, env_id, seed, gamma, index=0, update_stats=False
):
    def thunk():
        env = gym.make(env_id)
        if "FlattenObservation" in wrappers:
            env = gym.wrappers.FlattenObservation(env)
        if "RecordEpisodeStatistics" in wrappers:
            env = gym.wrappers.RecordEpisodeStatistics(env)
        if "ClipAction" in wrappers:
            env = gym.wrappers.ClipAction(env)
        if "NormalizeObservation" in wrappers:
            env = gym.wrappers.NormalizeObservation(env)
            env.update_running_mean = update_stats
            obs_rms = RunningMeanStd(
                shape=env.observation_space.shape,
                dtype=env.observation_space.dtype,
            )
            nbr_saved_envparams = envparams["obs_means"].shape[0]
            obs_rms.mean = envparams["obs_means"][index % nbr_saved_envparams]
            obs_rms.var = envparams["obs_vars"][index % nbr_saved_envparams]
            obs_rms.count = envparams["obs_counts"][index % nbr_saved_envparams]
            env.set_wrapper_attr("obs_rms", obs_rms)
        if "TransformObservation" in wrappers:
            range_obs = envparams["range_obs"]
            env = gym.wrappers.TransformObservation(
                env, lambda obs: np.clip(obs, range_obs[0], range_obs[-1]), env.observation_space
            )

        if "NormalizeReward" in wrappers:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env.update_running_mean = update_stats
            return_rms = RunningMeanStd(shape=())
            nbr_saved_envparams = envparams["return_means"].shape[0]
            return_rms.mean = envparams["return_means"][index % nbr_saved_envparams]
            return_rms.var = envparams["return_vars"][index % nbr_saved_envparams]
            return_rms.count = envparams["return_counts"][index % nbr_saved_envparams]
            env.set_wrapper_attr("return_rms", return_rms)
            env.discounted_reward = envparams["return_discounted_reward"][
                index % nbr_saved_envparams
            ]
            env.gamma = envparams["return_gamma"][index % nbr_saved_envparams]
            env.epsilon = envparams["returns_epsilon"][index % nbr_saved_envparams]
        if "TransformReward" in wrappers:
            range_reward = envparams["range_reward"]
            env = gym.wrappers.TransformReward(
                env, lambda reward: np.clip(reward, range_reward[0], range_reward[-1])
            )
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def make_env_mujoco(
    env_id,
    seed,
    gamma,
    norm_obs=True,
    clip_obs=True,
    range_obs=(-10, 10),
    norm_reward=True,
    clip_reward=True,
    range_reward=(-10, 10),
    update_stats=True,
):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        if norm_obs:
            env = gym.wrappers.NormalizeObservation(env)
            env.update_running_mean = update_stats
        if clip_obs:
            assert isinstance(range_obs, (tuple, list))
            env = gym.wrappers.TransformObservation(
                env, lambda obs: np.clip(obs, range_obs[0], range_obs[-1]), env.observation_space
            )
        if norm_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env.update_running_mean = update_stats
        if clip_reward:
            assert isinstance(range_reward, (tuple, list))
            env = gym.wrappers.TransformReward(
                env, lambda reward: np.clip(reward, range_reward[0], range_reward[-1])
            )
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
