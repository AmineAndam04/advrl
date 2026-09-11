import copy
import datetime
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import flatdim
from ml_collections import ConfigDict

from advrl.agents import NETWORK_REGISTRY
from advrl.envs.mujoco import make_env_mujoco
from advrl.envs.vec_env import VecEnv
from advrl.utils.envs import env_bounds
from advrl.utils.logger import Logger
from advrl.utils.misc import set_device, set_random_seed
from advrl.utils.save import env_metadata


class Base_MLP_Trainer:
    def __init__(self, config, workdir="runs", command=""):
        self.config = config
        self.workdir = workdir
        self.command = command

    def _setup(self):
        """
        Prepare the environment, optimizers, seeds ...
        """
        set_random_seed(self.config.seed)
        self.env = VecEnv(
            [
                make_env_mujoco(
                    env_id=self.config.env_id,
                    seed=self.config.seed + i,
                    gamma=self.config.gamma,
                    norm_obs=self.config.norm_obs,
                    clip_obs=self.config.clip_obs,
                    range_obs=self.config.range_obs,
                    norm_reward=self.config.norm_reward,
                    clip_reward=self.config.clip_reward,
                    range_reward=self.config.range_reward,
                    update_stats=True,
                )
                for i in range(self.config.num_envs)
            ]
        )
        if self.config.num_eval_envs > 0:
            self.eval_env = VecEnv(
                [
                    make_env_mujoco(
                        env_id=self.config.env_id,
                        seed=self.config.seed + 50 + i,
                        gamma=self.config.gamma,
                        norm_obs=self.config.norm_obs,
                        clip_obs=self.config.clip_obs,
                        range_obs=self.config.range_obs,
                        norm_reward=False,
                        clip_reward=False,
                        range_reward=self.config.range_reward,
                        update_stats=False,
                    )
                    for i in range(self.config.num_eval_envs)
                ]
            )
        # Lower and upper bound of observations
        self.low, self.high = env_bounds(
            env=self.env, clip_obs=self.config.clip_obs, range_obs=self.config.range_obs
        )
        self.device = set_device(self.config.device)
        # Setup the networks
        self.actor = NETWORK_REGISTRY[self.config.actor](
            in_dim=flatdim(self.env.single_observation_space),
            out_dim=flatdim(self.env.single_action_space),
            net_arch=ConfigDict(
                {
                    "pi_net": self.config.pi_net,
                    "act_fun": self.config.pi_act_fun,
                    "ortho_init": self.config.ortho_init,
                    "orth_std": self.config.orth_std,
                    "log_std_init": self.config.log_std_init,
                }
            ),
        ).to(self.device)
        self.critic = NETWORK_REGISTRY[self.config.critic](
            in_dim=flatdim(self.env.single_observation_space),
            net_arch=ConfigDict(
                {
                    "vf_net": self.config.vf_net,
                    "act_fun": self.config.vf_act_fun,
                    "ortho_init": self.config.ortho_init,
                    "orth_std": self.config.orth_std,
                }
            ),
        ).to(self.device)
        # Optimizers
        optimizer = getattr(optim, self.config.optim)
        self.actor_optimizer = optimizer(self.actor.parameters(), lr=self.config.lr_actor)
        self.critic_optimizer = optimizer(self.critic.parameters(), lr=self.config.lr_critic)
        # Trackers
        self.current_step, self.num_episodes = 0, 0
        self.ep_rewards, self.ep_lengths = [], []
        self.eval_flag = False
        # Logger
        time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_name = f"{self.config.prefix_run_name}-{self.config.exp_name}__{time_token}"
        self.logger = Logger(
            mode=self.config.logger,
            wand_project=self.config.project_name,
            run_name=self.run_name,
            log_dir=f"{self.workdir}/{self.config.env_id}/{self.config.algo_name}",
            config=self.config,
        )
        # Save config and command
        self.path = f"{self.workdir}/{self.config.env_id}/{self.config.algo_name}/{self.run_name}"
        os.makedirs(self.path, exist_ok=True)
        with open(f"{self.path}/config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        with open(f"{self.path}/command.txt", "w") as f:
            f.write("python " + " ".join(self.command) + "\n")

    def collect_nsteps(
        self,
    ):
        if self.current_step == 0:
            observation, _ = self.env.reset()
            observation = torch.as_tensor(observation).to(self.device)
        else:
            observation = self.last_obs
        obs = torch.zeros(
            (self.config.n_steps, self.config.num_envs) + self.env.single_observation_space.shape
        ).to(self.device)
        actions = torch.zeros(
            (self.config.n_steps, self.config.num_envs) + self.env.single_action_space.shape
        ).to(self.device)
        logprobs = torch.zeros((self.config.n_steps, self.config.num_envs)).to(self.device)
        rewards = torch.zeros((self.config.n_steps, self.config.num_envs)).to(self.device)
        dones = torch.zeros((self.config.n_steps, self.config.num_envs)).to(self.device)
        values = torch.zeros((self.config.n_steps, self.config.num_envs)).to(self.device)
        for step in range(0, self.config.n_steps):
            with torch.no_grad():
                action, logprob, *_ = self.actor(obs=observation)
                val, _ = self.critic(obs=observation)
            next_obs, reward, terminations, truncations, infos = self.env.step(action.cpu().numpy())
            done = np.logical_or(terminations, truncations)
            done = torch.as_tensor(done).to(self.device)
            obs[step] = observation
            rewards[step] = torch.as_tensor(reward).to(self.device).view(-1)
            dones[step] = done
            actions[step] = action
            values[step] = val.flatten()
            logprobs[step] = logprob
            observation = torch.as_tensor(next_obs).to(self.device)
            self.current_step += self.config.num_envs
            for info in infos:
                if "episode" in info:
                    self.num_episodes += 1
                    ep_rd = info["episode"]["r"]
                    ep_len = info["episode"]["l"]
                    self.ep_rewards.append(ep_rd)
                    self.ep_lengths.append(ep_len)
                    if len(self.ep_rewards) == self.config.log_every:
                        self.logger.log_metrics(
                            metrics={
                                "rollout/ep_reward": np.mean(self.ep_rewards),
                                "rollout/ep_length": np.mean(self.ep_lengths),
                            },
                            step=self.current_step,
                        )
                        self.ep_rewards = []
                        self.ep_lengths = []
                    if (
                        self.config.num_eval_envs > 0
                        and self.config.eval_every > 0
                        and self.num_episodes % self.config.eval_every == 0
                    ):
                        self.eval_flag = True

        self.last_obs = observation
        with torch.no_grad():
            last_value, _ = self.critic(self.last_obs)
        returns, advantages = self.compute_advantages_and_returns(values, last_value, rewards, dones)
        b_obs = obs.reshape((-1,) + self.env.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + self.env.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_rewards = rewards.reshape(-1)
        b_dones = dones.reshape(-1)
        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_rewards, b_dones

    def compute_advantages_and_returns(self, values, last_value, rewards, dones, config=None):
        config = config or self.config
        next_value = last_value.reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(self.device)
        lastgaelam = 0
        next_values = torch.cat([values[1:], next_value], dim=0)
        for t in reversed(range(config.n_steps)):
            nextnonterminal = 1.0 - dones[t]
            delta = rewards[t] + config.gamma * next_values[t] * nextnonterminal - values[t]
            advantages[t] = lastgaelam = (
                delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            )
        returns = advantages + values
        return returns, advantages

    def update_lr(self):
        frac = 1.0 - (self.current_step - 1.0) / self.config.total_timesteps
        lr_actor_now = max(self.config.min_lr, frac * self.config.lr_actor)
        lr_critic_now = max(self.config.min_lr, frac * self.config.lr_critic)
        self.actor_optimizer.param_groups[0]["lr"] = lr_actor_now
        self.critic_optimizer.param_groups[0]["lr"] = lr_critic_now
        self.logger.log_metrics(
            metrics={
                "train/lr_actor": lr_actor_now,
                "train/lr_critic": lr_critic_now,
            },
            step=self.current_step,
        )

    def update_critic(self, mb_obs, mb_returns, mb_values):
        newvalue, _ = self.critic(obs=mb_obs)
        newvalue = newvalue.view(-1)
        if self.config.clip_range_vf > 0:
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_clipped = mb_values + torch.clamp(
                newvalue - mb_values,
                -self.config.clip_range_vf,
                self.config.clip_range_vf,
            )
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

        self.critic_optimizer.zero_grad()
        v_loss.backward()
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_optimizer.step()
        return v_loss.item()

    def evaluate(self):
        if self.config.norm_obs:
            for i in range(self.config.num_eval_envs):
                self.eval_env.envs[i].set_wrapper_attr(
                    "obs_rms",
                    copy.deepcopy(self.env.envs[i % self.config.num_envs].get_wrapper_attr("obs_rms")),
                )
        eval_rewards = []
        eval_lengths = []
        obs, _ = self.eval_env.reset()
        while len(eval_rewards) < self.config.num_eval_episodes:
            obs = torch.as_tensor(obs).to(self.device)
            with torch.no_grad():
                action, *_ = self.actor.get_action(obs=obs, deterministic=self.config.deterministic)
            obs, _, _, _, infos = self.eval_env.step(action.cpu().numpy())
            for info in infos:
                if "episode" in info:
                    ep_rd = info["episode"]["r"]
                    ep_len = info["episode"]["l"]
                    eval_rewards.append(ep_rd)
                    eval_lengths.append(ep_len)
        self.logger.log_metrics(
            metrics={
                "eval/ep_reward": np.mean(eval_rewards),
                "eval/ep_length": np.mean(eval_lengths),
            },
            step=self.current_step,
        )

    def save(self):
        torch.save(self.actor.state_dict(), f"{self.path}/policy.pt")
        torch.save(self.critic.state_dict(), f"{self.path}/critic.pt")
        envparams, wrappers = env_metadata(env=self.env, config=self.config)
        with open(f"{self.path}/wrappers.json", "w") as f:
            json.dump(wrappers, f, indent=2)
        np.savez_compressed(f"{self.path}/envparams.npz", **envparams)
