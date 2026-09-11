import gymnasium as gym
from gymnasium.spaces import flatdim
import torch


def get_obs_and_act_dim(env_id):
    env = gym.make(env_id)
    obs_dim = flatdim(env.observation_space)
    act_dim = flatdim(env.action_space)
    return obs_dim, act_dim


def env_bounds(env, clip_obs, range_obs):
    low = env.single_observation_space.low
    high = env.single_observation_space.high
    if clip_obs:
        low[low < range_obs[0]] = range_obs[0]
        high[high > range_obs[1]] = range_obs[1]
    return torch.from_numpy(low), torch.from_numpy(high)
