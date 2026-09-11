import json
import torch
import numpy as np
from ml_collections import ConfigDict
import os


def env_metadata(env, config):
    wrappers = [wrapper.name for wrapper in env.envs[0].spec.additional_wrappers]
    metadata = {}
    if "NormalizeObservation" in wrappers:
        obs_means = []
        obs_vars = []
        obs_counts = []
        for i in range(len(env.envs)):
            obs_rms = env.envs[i].get_wrapper_attr("obs_rms")
            obs_means.append(obs_rms.mean)
            obs_vars.append(obs_rms.var)
            obs_counts.append(obs_rms.count)
        metadata.update(
            {
                "obs_means": np.stack(obs_means),
                "obs_vars": np.stack(obs_vars),
                "obs_counts": np.stack(obs_counts),
            }
        )

    if "NormalizeReward" in wrappers:
        return_means = []
        return_vars = []
        return_counts = []
        return_discounted_reward = []
        return_gamma = []
        returns_epsilon = []
        for i in range(len(env.envs)):
            return_rms = env.envs[i].get_wrapper_attr("return_rms")
            return_means.append(return_rms.mean)
            return_vars.append(return_rms.var)
            return_counts.append(return_rms.count)
            return_discounted_reward.append(env.envs[i].get_wrapper_attr("discounted_reward"))
            return_gamma.append(env.envs[i].get_wrapper_attr("gamma"))
            returns_epsilon.append(env.envs[i].get_wrapper_attr("epsilon"))

        return_means = np.stack(return_means)
        return_vars = np.stack(return_vars)
        return_counts = np.stack(return_counts)
        return_discounted_reward = np.stack(return_discounted_reward)
        return_gamma = np.stack(return_gamma)
        returns_epsilon = np.stack(returns_epsilon)

        metadata.update(
            {
                "return_means": np.stack(return_means),
                "return_vars": np.stack(return_vars),
                "return_counts": np.stack(return_counts),
                "return_discounted_reward": np.stack(return_discounted_reward),
                "return_gamma": np.stack(return_gamma),
                "returns_epsilon": np.stack(returns_epsilon),
            }
        )
    if "TransformObservation" in wrappers:
        metadata.update({"range_obs": config.range_obs})
    if "TransformReward" in wrappers:
        metadata.update({"range_reward": config.range_reward})

    return metadata, wrappers


def load_obs_rms(path, env):
    envparams = np.load(f"{path}/envparams.npz")
    obs_means = envparams["obs_means"]
    obs_vars = envparams["obs_vars"]
    obs_counts = envparams["obs_counts"]

    for i in range(env.num_envs):
        temp_obs_rms = env.envs[i].get_wrapper_attr("obs_rms")
        temp_obs_rms.mean = obs_means[i % obs_means.shape[0]]
        temp_obs_rms.var = obs_vars[i % obs_means.shape[0]]
        temp_obs_rms.count = obs_counts[i % obs_means.shape[0]]
        env.envs[i].set_wrapper_attr("obs_rms", temp_obs_rms)
    return env


def load_saved_folder(path):
    with open(f"{path}/config.json") as f:
        config = ConfigDict(json.load(f))
    with open(f"{path}/wrappers.json") as f:
        wrappers = json.load(f)
    envparams = np.load(f"{path}/envparams.npz")
    state_dicts = {}
    if os.path.isfile(f"{path}/policy.pt"):
        state_dicts.update({"policy": torch.load(f"{path}/policy.pt", map_location="cpu")})
    if os.path.isfile(f"{path}/critic.pt"):
        state_dicts.update({"critic": torch.load(f"{path}/critic.pt", map_location="cpu")})
    return config, wrappers, envparams, state_dicts
