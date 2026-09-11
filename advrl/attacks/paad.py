import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gymnasium.spaces import flatdim
from torch.distributions.normal import Normal
from advrl.utils.misc import set_random_seed, set_scheduler
from advrl.envs.mujoco import create_make_env_mujoco_from_json
from advrl.envs.vec_env import VecEnv, EpisodicVecEnv
from ml_collections import ConfigDict
from advrl.agents import NETWORK_REGISTRY


class PAADAttack:
    """
    PA-AD attack
    Paper: Who Is the Strongest Enemy? Towards Optimal and Efficient Evasion Attacks in Deep RL
    Arxiv: https://arxiv.org/abs/2106.05087
    """

    def __init__(self, config, agent, low, high, **kwargs):
        self.config = config
        self.agent = agent
        self.num_iter = self.config.num_iter
        self.eps = self.config.eps
        self.step_size = self.config.step_size
        if self.step_size < 0:
            self.step_size = self.eps / self.num_iter
        self.low = low
        self.high = high
        self.device = config.device
        if not config.train:
            self.load()
        self.hidden_paad = None

    def reset_hidden(self, index):
        if self.hidden_paad is not None:
            self.hidden_paad[0][:, index] = 0
            self.hidden_paad[1][:, index] = 0

    def _setup_mlp(self):
        self.env = VecEnv(
            [
                create_make_env_mujoco_from_json(
                    wrappers=self.wrappers,
                    envparams=self.envparams,
                    env_id=self.cfg_train.env_id,
                    seed=self.cfg_train.seed + i,
                    gamma=self.cfg_train.gamma,
                    index=i,
                    update_stats=False,
                )
                for i in range(self.cfg_train.num_envs)
            ]
        )
        self.actor = NETWORK_REGISTRY[self.cfg_train.actor](
            in_dim=flatdim(self.env.single_observation_space),
            out_dim=flatdim(self.env.single_action_space),
            net_arch=ConfigDict(
                {
                    "pi_net": self.cfg_train.pi_net,
                    "act_fun": self.cfg_train.pi_act_fun,
                    "ortho_init": self.cfg_train.ortho_init,
                    "orth_std": self.cfg_train.orth_std,
                    "log_std_init": self.cfg_train.log_std_init,
                },
            ),
        ).to(self.device)
        self.critic = NETWORK_REGISTRY[self.cfg_train.critic](
            in_dim=flatdim(self.env.single_observation_space),
            net_arch=ConfigDict(
                {
                    "vf_net": self.cfg_train.vf_net,
                    "act_fun": self.cfg_train.vf_act_fun,
                    "ortho_init": self.cfg_train.ortho_init,
                    "orth_std": self.cfg_train.orth_std,
                }
            ),
        ).to(self.device)

    def _setup_lstm(self):
        self.env = EpisodicVecEnv(
            [
                create_make_env_mujoco_from_json(
                    wrappers=self.wrappers,
                    envparams=self.envparams,
                    env_id=self.cfg_train.env_id,
                    seed=self.cfg_train.seed + i,
                    gamma=self.cfg_train.gamma,
                    index=i,
                    update_stats=False,
                )
                for i in range(self.cfg_train.num_envs)
            ]
        )
        # Setup the networks
        self.actor = NETWORK_REGISTRY[self.cfg_train.actor](
            in_dim=flatdim(self.env.single_observation_space),
            out_dim=flatdim(self.env.single_action_space),
            net_arch=ConfigDict(
                {
                    "pi_lstm_dim": self.cfg_train.pi_net[0],
                    "ortho_init": self.cfg_train.ortho_init,
                    "orth_std": self.cfg_train.orth_std,
                    "log_std_init": self.cfg_train.log_std_init,
                }
            ),
        ).to(self.device)
        self.critic = NETWORK_REGISTRY[self.cfg_train.critic](
            in_dim=flatdim(self.env.single_observation_space),
            net_arch=ConfigDict(
                {
                    "vf_net": self.cfg_train.vf_net,
                    "vf_lstm_dim": self.cfg_train.vf_net[-1],
                    "ortho_init": self.cfg_train.ortho_init,
                    "orth_std": self.cfg_train.orth_std,
                }
            ),
        ).to(self.device)

    def _setup(self):
        set_random_seed(self.cfg_train.seed)

        if self.cfg_train.actor_lstm:
            self._setup_lstm()
        else:
            self._setup_mlp()

        # Optimizers
        optimizer = getattr(optim, self.cfg_train.optim)
        self.actor_optimizer = optimizer(self.actor.parameters(), lr=self.cfg_train.lr_actor)
        self.critic_optimizer = optimizer(self.critic.parameters(), lr=self.cfg_train.lr_critic)

        self.eps_schedule = set_scheduler(config=self.cfg_train)

        os.makedirs(self.save_path, exist_ok=True)
        self.run_name = "PAAD"
        if self.cfg_train.actor_lstm:
            self.run_name += "-LSTM"
        else:
            self.run_name += "-MLP"
        self.logger.run_name = self.run_name
        self.logger.initialize_logger()

    def paad_perturbation(
        self, observation, eps, mu_dir, std_dir, step_size, num_iter, deterministic_victim, h=None
    ):
        if step_size < 0:
            step_size = eps / num_iter
        else:
            step_size = step_size
        delta = torch.zeros_like(observation).uniform_(-step_size, step_size)
        for _ in range(num_iter):
            delta = delta.clone().detach().requires_grad_(True)
            mu, std, *_ = self.agent.get_mu_and_std(obs=observation + delta, h=h)
            mu = mu.reshape_as(mu_dir)
            std = std.reshape_as(std_dir)
            if deterministic_victim:
                loss = ((mu - mu_dir) ** 2).sum()
            else:
                loss = torch.distributions.kl.kl_divergence(
                    p=Normal(mu, std), q=Normal(mu_dir, std_dir)
                ).sum()
            loss.backward()
            grad = delta.grad
            delta = torch.clamp(delta - step_size * torch.sign(grad), -eps, eps)
        delta = delta.clone().detach()
        return torch.clamp(observation + delta, min=self.low, max=self.high)

    def mlp_collect_nsteps(
        self,
    ):
        if self.current_step == 0:
            observation, _ = self.env.reset()
            observation = torch.Tensor(observation).to(self.device)
            self.agent_h = None
        else:
            observation = self.last_obs

        obs = torch.zeros(
            (self.cfg_train.n_steps, self.cfg_train.num_envs)
            + self.env.single_observation_space.shape
        ).to(self.device)
        actions = torch.zeros(
            (self.cfg_train.n_steps, self.cfg_train.num_envs) + self.env.single_action_space.shape
        ).to(self.device)
        logprobs = torch.zeros((self.cfg_train.n_steps, self.cfg_train.num_envs)).to(self.device)
        rewards = torch.zeros((self.cfg_train.n_steps, self.cfg_train.num_envs)).to(self.device)
        dones = torch.zeros((self.cfg_train.n_steps, self.cfg_train.num_envs)).to(self.device)
        values = torch.zeros((self.cfg_train.n_steps, self.cfg_train.num_envs)).to(self.device)
        eps = self.eps_schedule(current_t=self.current_step / self.cfg_train.total_timesteps)
        for step in range(0, self.cfg_train.n_steps):
            with torch.no_grad():
                mu_dir, std_dir, *_ = self.actor.get_mu_and_std(obs=observation)
                probs = Normal(mu_dir, std_dir)
                action = probs.sample()
                logprob = probs.log_prob(action).sum(-1)
            adv_obs = self.paad_perturbation(
                observation=observation,
                eps=eps,
                mu_dir=action,
                std_dir=std_dir,
                step_size=self.cfg_train.step_size,
                num_iter=self.cfg_train.num_iter,
                deterministic_victim=self.config.deterministic,
                h=self.agent_h,
            )
            with torch.no_grad():
                agent_action, self.agent_h = self.agent.get_action(
                    obs=adv_obs, deterministic=self.config.deterministic, h=self.agent_h
                )
                val, _ = self.critic(obs=observation)
            agent_action = agent_action.reshape(
                (self.cfg_train.num_envs,) + self.env.single_action_space.shape
            )
            next_obs, reward, terminations, truncations, infos = self.env.step(
                agent_action.cpu().numpy()
            )
            done = np.logical_or(terminations, truncations)
            done = torch.Tensor(done).to(self.device)
            obs[step] = observation
            rewards[step] = -torch.tensor(reward).to(self.device).view(-1)
            dones[step] = done
            actions[step] = action
            values[step] = val.flatten()
            logprobs[step] = logprob
            observation = torch.Tensor(next_obs).to(self.device)
            self.current_step += self.cfg_train.num_envs
            for i, info in enumerate(infos):
                if "episode" in info:
                    if self.agent_h is not None:
                        self.agent_h[0][:, i] = 0
                        self.agent_h[1][:, i] = 0
                    ep_rd = info["episode"]["r"]
                    ep_len = info["episode"]["l"]
                    self.traj_rewards.append(ep_rd)
                    self.traj_lengths.append(ep_len)
                    if len(self.traj_rewards) == self.cfg_train.log_every:
                        self.logger.log_metrics(
                            metrics={
                                "traj/rewards": np.mean(self.traj_rewards),
                                "traj/traj_lengths": np.mean(self.traj_lengths),
                                "traj/eps": eps,
                            },
                            step=self.current_step,
                        )
                        self.traj_rewards = []
                        self.traj_lengths = []

        self.last_obs = observation

        returns, advantages = self.mlp_compute_advantages_and_returns(
            values, self.last_obs, rewards, dones
        )
        b_obs = obs.reshape((-1,) + self.env.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + self.env.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values

    def mlp_compute_advantages_and_returns(self, values, last_obs, rewards, dones):
        with torch.no_grad():
            next_value, _ = self.critic(last_obs)
            next_value = next_value.reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.cfg_train.n_steps)):
                if t == self.cfg_train.n_steps - 1:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = values[t + 1]
                delta = rewards[t] + self.cfg_train.gamma * nextvalues * nextnonterminal - values[t]

                advantages[t] = lastgaelam = (
                    delta
                    + self.cfg_train.gamma
                    * self.cfg_train.gae_lambda
                    * nextnonterminal
                    * lastgaelam
                )
            returns = advantages + values
        return returns, advantages

    def lstm_collect_nsteps(
        self,
    ):
        observation, _ = self.env.reset()
        observation = torch.Tensor(observation).to(self.device)
        alive_envs = self.env.alive_envs
        episodes = [
            {
                "obs": [],
                "actions": [],
                "hidden_h0": [],
                "hidden_c0": [],
                "logprobs": [],
                "rewards": [],
                "values": [],
            }
            for _ in range(self.cfg_train.num_envs)
        ]
        if self.cfg_train.critic_lstm:
            for episode in episodes:
                episode.update(
                    {
                        "hidden_vf_h0": [],
                        "hidden_vf_c0": [],
                    }
                )
            h_vf = (
                torch.zeros(1, self.cfg_train.num_envs, self.cfg_train.vf_net[-1]),
                torch.zeros(1, self.cfg_train.num_envs, self.cfg_train.vf_net[-1]),
            )
        else:
            h_vf = None
        h = (
            torch.zeros(1, self.cfg_train.num_envs, self.cfg_train.pi_net[0]),
            torch.zeros(1, self.cfg_train.num_envs, self.cfg_train.pi_net[0]),
        )
        agent_h = None
        eps = self.eps_schedule(current_t=self.current_step / self.cfg_train.total_timesteps)
        while len(alive_envs) > 0:
            with torch.no_grad():
                mu_dir, std_dir, next_h = self.actor.get_mu_and_std(
                    obs=observation.unsqueeze(1), h=h
                )
                mu_dir = mu_dir.reshape((len(alive_envs),) + self.env.single_action_space.shape)
                std_dir = std_dir.reshape((len(alive_envs),) + self.env.single_action_space.shape)
                probs = Normal(mu_dir, std_dir)
                action = probs.sample()
                logprob = probs.log_prob(action).sum(-1)
            adv_obs = self.paad_perturbation(
                observation=observation,
                eps=eps,
                mu_dir=action,
                std_dir=std_dir,
                step_size=self.cfg_train.step_size,
                num_iter=self.cfg_train.num_iter,
                h=agent_h,
                deterministic_victim=self.config.deterministic,
            )
            with torch.no_grad():
                agent_action, agent_h = self.agent.get_action(
                    obs=adv_obs.unsqueeze(1),
                    h=agent_h,
                    deterministic=self.config.deterministic,
                )
                agent_action = agent_action.reshape(
                    (len(alive_envs),) + self.env.single_action_space.shape
                )
                value, next_h_vf = self.critic(obs=observation.unsqueeze(1), h=h_vf)
            next_obs, reward, terminated, truncated, infos = self.env.step(
                agent_action.cpu().numpy()
            )
            reward = -torch.tensor(reward).to(self.device).view(-1)
            value = value.flatten()
            for i, j in enumerate(alive_envs):
                episodes[j]["obs"].append(observation[i])
                episodes[j]["actions"].append(action[i])
                episodes[j]["logprobs"].append(logprob[i])
                episodes[j]["rewards"].append(reward[i])
                episodes[j]["values"].append(value[i])
                episodes[j]["hidden_h0"].append(h[0][:, i])
                episodes[j]["hidden_c0"].append(h[1][:, i])
                if self.cfg_train.critic_lstm:
                    episodes[j]["hidden_vf_h0"].append(h_vf[0][:, i])
                    episodes[j]["hidden_vf_c0"].append(h_vf[1][:, i])

            self.current_step += len(alive_envs)
            alive_envs = self.env.alive_envs
            done = np.logical_or(terminated, truncated)
            observation = torch.Tensor(next_obs).to(self.device)
            observation = observation[~done]
            h = (next_h[0][:, ~done], next_h[1][:, ~done])
            if agent_h is not None:
                agent_h = (agent_h[0][:, ~done], agent_h[1][:, ~done])
            if self.cfg_train.critic_lstm:
                h_vf = (next_h_vf[0][:, ~done], next_h_vf[1][:, ~done])
            for info in infos:
                if "episode" in info:
                    ep_rd = info["episode"]["r"]
                    ep_len = info["episode"]["l"]
                    self.traj_rewards.append(ep_rd)
                    self.traj_lengths.append(ep_len)
                    if len(self.traj_rewards) == self.cfg_train.log_every:
                        self.logger.log_metrics(
                            metrics={
                                "traj/rewards": np.mean(self.traj_rewards),
                                "traj/traj_lengths": np.mean(self.traj_lengths),
                                "traj/eps": eps,
                            },
                            step=self.current_step,
                        )
                        self.traj_rewards = []
                        self.traj_lengths = []

        (
            b_obs,
            hidden_h0,
            hidden_c0,
            b_actions,
            b_logprobs,
            b_rewards,
            b_values,
            hidden_vf_h0,
            hidden_vf_c0,
            b_mask,
        ) = self.batchify(episodes)
        b_advantages, b_returns = self.lstm_compute_advantages_and_returns(
            b_rewards=b_rewards, b_values=b_values
        )
        return (
            b_obs,
            hidden_h0,
            hidden_c0,
            b_logprobs,
            b_actions,
            b_advantages,
            b_returns,
            b_values,
            hidden_vf_h0,
            hidden_vf_c0,
            b_mask,
        )

    def batchify(self, episodes):
        for episode in episodes:
            for key, vals in episode.items():
                episode[key] = torch.stack(vals).float().to(self.device)
        lengths = [len(episode["obs"]) for episode in episodes]
        max_length = max(lengths)
        obs = torch.zeros(
            (self.cfg_train.num_envs, max_length) + self.env.single_observation_space.shape
        ).to(self.device)
        actions = torch.zeros(
            (self.cfg_train.num_envs, max_length) + self.env.single_action_space.shape
        ).to(self.device)
        logprobs = torch.zeros((self.cfg_train.num_envs, max_length)).to(self.device)
        rewards = torch.zeros((self.cfg_train.num_envs, max_length)).to(self.device)
        values = torch.zeros((self.cfg_train.num_envs, max_length)).to(self.device)
        hidden_h0 = torch.zeros(
            (max_length, 1, self.cfg_train.num_envs, self.cfg_train.pi_net[0])
        ).to(self.device)
        hidden_c0 = torch.zeros(
            (max_length, 1, self.cfg_train.num_envs, self.cfg_train.pi_net[0])
        ).to(self.device)
        if self.cfg_train.critic_lstm:
            hidden_vf_h0 = torch.zeros(
                (max_length, 1, self.cfg_train.num_envs, self.cfg_train.vf_net[-1])
            ).to(self.device)
            hidden_vf_c0 = torch.zeros(
                (max_length, 1, self.cfg_train.num_envs, self.cfg_train.vf_net[-1])
            ).to(self.device)
        else:
            hidden_vf_h0, hidden_vf_c0 = None, None
        mask = torch.zeros((self.cfg_train.num_envs, max_length)).to(self.device)
        for i in range(self.cfg_train.num_envs):
            obs[i, : lengths[i]] = episodes[i]["obs"]
            actions[i, : lengths[i]] = episodes[i]["actions"]
            logprobs[i, : lengths[i]] = episodes[i]["logprobs"]
            rewards[i, : lengths[i]] = episodes[i]["rewards"]
            values[i, : lengths[i]] = episodes[i]["values"]
            mask[i, : lengths[i]] = 1
            hidden_h0[: lengths[i], :, i] = episodes[i]["hidden_h0"]
            hidden_c0[: lengths[i], :, i] = episodes[i]["hidden_c0"]
            if self.cfg_train.critic_lstm:
                hidden_vf_h0[: lengths[i], :, i] = episodes[i]["hidden_vf_h0"]
                hidden_vf_c0[: lengths[i], :, i] = episodes[i]["hidden_vf_c0"]
        return (
            obs,
            hidden_h0,
            hidden_c0,
            actions,
            logprobs,
            rewards,
            values,
            hidden_vf_h0,
            hidden_vf_c0,
            mask.bool(),
        )

    def lstm_compute_advantages_and_returns(self, b_rewards, b_values):
        advantages = torch.zeros_like(b_rewards).to(self.device)
        lastgaelam = 0
        for t in reversed(range(b_rewards.shape[1])):
            if t == (b_rewards.shape[1] - 1):
                next_value = 0
            else:
                next_value = b_values[:, t + 1]
            delta = b_rewards[:, t] + self.cfg_train.gamma * next_value - b_values[:, t]
            advantages[:, t] = lastgaelam = (
                delta + self.cfg_train.gamma * self.cfg_train.gae_lambda * lastgaelam
            )
        returns = advantages + b_values
        return advantages, returns

    def lstm_train(self):
        self.traj_rewards = []
        self.traj_lengths = []
        self.current_step = 0
        while self.current_step < self.cfg_train.total_timesteps:

            (
                b_obs,
                b_hidden_h0,
                b_hidden_c0,
                b_logprobs,
                b_actions,
                b_advantages,
                b_returns,
                b_values,
                b_hidden_vf_h0,
                b_hidden_vf_c0,
                b_mask,
            ) = self.lstm_collect_nsteps()
            if self.cfg_train.normalize_advantage:
                b_advantages = (b_advantages - b_advantages[b_mask].mean()) / (
                    b_advantages[b_mask].std() + 1e-8
                )
            if self.cfg_train.normalize_return:
                b_returns = (b_returns - b_returns[b_mask].mean()) / (
                    b_returns[b_mask].std() + 1e-8
                )
            if self.cfg_train.anneal_lr:
                frac = 1.0 - (self.current_step - 1.0) / self.cfg_train.total_timesteps
                lr_actor_now = max(self.cfg_train.min_lr, frac * self.cfg_train.lr_actor)
                lr_critic_now = max(self.cfg_train.min_lr, frac * self.cfg_train.lr_critic)
                self.actor_optimizer.param_groups[0]["lr"] = lr_actor_now
                self.critic_optimizer.param_groups[0]["lr"] = lr_critic_now
            v_losses, pg_losses, entropy_losses, approx_kls, clipfracs = (
                [],
                [],
                [],
                [],
                [],
            )
            for _ in range(self.cfg_train.n_epochs):
                for start in range(0, b_obs.shape[1], self.cfg_train.batch_size):
                    end = start + self.cfg_train.batch_size
                    h = (b_hidden_h0[start], b_hidden_c0[start])

                    _, newlogprob, entropy, *_ = self.actor(
                        obs=b_obs[:, start:end], h=h, action=b_actions[:, start:end]
                    )
                    logratio = newlogprob - b_logprobs[:, start:end]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio)[b_mask[:, start:end]].mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.cfg_train.clip_range)
                            .float()[b_mask[:, start:end]]
                            .mean()
                            .item()
                        ]
                        approx_kls.append(approx_kl.item())
                    mb_advantages = b_advantages[:, start:end]
                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.cfg_train.clip_range, 1 + self.cfg_train.clip_range
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2)[b_mask[:, start:end]].mean()
                    pg_losses.append(pg_loss.item())
                    entropy_loss = entropy[b_mask[:, start:end]].mean()
                    entropy_losses.append(entropy_loss.item())
                    loss = pg_loss - self.cfg_train.ent_coef * entropy_loss

                    # Value loss
                    if self.cfg_train.critic_lstm:
                        h_vf = (b_hidden_vf_h0[start], b_hidden_vf_c0[start])
                    else:
                        h_vf = None
                    newvalue, _ = self.critic(obs=b_obs[:, start:end], h=h_vf)
                    newvalue = newvalue.reshape(self.cfg_train.num_envs, -1)
                    if self.cfg_train.clip_range_vf > 0:
                        v_loss_unclipped = (newvalue - b_returns[:, start:end]) ** 2
                        v_clipped = b_values[:, start:end] + torch.clamp(
                            newvalue - b_values[:, start:end],
                            -self.cfg_train.clip_range_vf,
                            self.cfg_train.clip_range_vf,
                        )
                        v_loss_clipped = (v_clipped - b_returns[:, start:end]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max[b_mask[:, start:end]].mean()
                    else:
                        v_loss = (
                            0.5
                            * ((newvalue - b_returns[:, start:end]) ** 2)[
                                b_mask[:, start:end]
                            ].mean()
                        )
                    v_losses.append(v_loss.item())

                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    v_loss.backward()
                    loss.backward()
                    if self.cfg_train.max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(
                            self.actor.parameters(), self.cfg_train.max_grad_norm
                        )
                        nn.utils.clip_grad_norm_(
                            self.critic.parameters(), self.cfg_train.max_grad_norm
                        )
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()

            y_pred, y_true = (
                b_values[b_mask].cpu().numpy(),
                b_returns[b_mask].cpu().numpy(),
            )
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            self.logger.log_metrics(
                metrics={
                    "train/value_loss": np.mean(v_losses),
                    "train/policy_loss": np.mean(pg_losses),
                    "train/entropy": np.mean(entropy_losses),
                    "train/approx_kl": np.mean(approx_kls),
                    "train/clipfrac": np.mean(clipfracs),
                    "train/explained_variance": explained_var,
                },
                step=self.current_step,
            )
            if self.cfg_train.anneal_lr:
                self.logger.log_metrics(
                    metrics={
                        "train/lr_actor": lr_actor_now,
                        "train/lr_critic": lr_critic_now,
                    },
                    step=self.current_step,
                )
        self.save()

    def mlp_train(self):
        self.traj_rewards = []
        self.traj_lengths = []
        self.current_step = 0
        while self.current_step < self.cfg_train.total_timesteps:
            b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = (
                self.mlp_collect_nsteps()
            )
            if self.cfg_train.normalize_advantage:
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            if self.cfg_train.normalize_return:
                b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-8)
            if self.cfg_train.anneal_lr:
                frac = 1.0 - (self.current_step - 1.0) / self.cfg_train.total_timesteps
                lr_actor_now = max(self.cfg_train.min_lr, frac * self.cfg_train.lr_actor)
                lr_critic_now = max(self.cfg_train.min_lr, frac * self.cfg_train.lr_critic)
                self.actor_optimizer.param_groups[0]["lr"] = lr_actor_now
                self.critic_optimizer.param_groups[0]["lr"] = lr_critic_now
            b_inds = np.arange(self.cfg_train.num_envs * self.cfg_train.n_steps)
            v_losses, pg_losses, entropy_losses, approx_kls, clipfracs = [], [], [], [], []
            for _ in range(self.cfg_train.n_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, len(b_inds), self.cfg_train.batch_size):
                    end = start + self.cfg_train.batch_size
                    mb_inds = b_inds[start:end]
                    _, newlogprob, entropy, *_ = self.actor(
                        obs=b_obs[mb_inds], action=b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.cfg_train.clip_range).float().mean().item()
                        ]
                        approx_kls.append(approx_kl.item())
                    mb_advantages = b_advantages[mb_inds]
                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.cfg_train.clip_range, 1 + self.cfg_train.clip_range
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    pg_losses.append(pg_loss.item())
                    entropy_loss = entropy.mean()
                    entropy_losses.append(entropy_loss.item())
                    loss = pg_loss - self.cfg_train.ent_coef * entropy_loss
                    # Value loss
                    newvalue, _ = self.critic(obs=b_obs[mb_inds])
                    newvalue = newvalue.view(-1)
                    if self.cfg_train.clip_range_vf > 0:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.cfg_train.clip_range_vf,
                            self.cfg_train.clip_range_vf,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    v_losses.append(v_loss.item())

                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    v_loss.backward()
                    loss.backward()
                    if self.cfg_train.max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(
                            self.actor.parameters(), self.cfg_train.max_grad_norm
                        )
                        nn.utils.clip_grad_norm_(
                            self.critic.parameters(), self.cfg_train.max_grad_norm
                        )
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            self.logger.log_metrics(
                metrics={
                    "train/value_loss": np.mean(v_losses),
                    "train/policy_loss": np.mean(pg_losses),
                    "train/entropy": np.mean(entropy_losses),
                    "train/approx_kl": np.mean(approx_kls),
                    "train/clipfrac": np.mean(clipfracs),
                    "train/explained_variance": explained_var,
                },
                step=self.current_step,
            )
            if self.cfg_train.anneal_lr:
                self.logger.log_metrics(
                    metrics={
                        "train/lr_actor": lr_actor_now,
                        "train/lr_critic": lr_critic_now,
                    },
                    step=self.current_step,
                )
        self.save()

    def train(self, cfg_train, wrappers, envparams, workdir, save_path, logger):
        self.cfg_train = cfg_train
        self.wrappers = wrappers
        self.envparams = envparams
        self.workdir = workdir
        self.save_path = save_path
        self.logger = logger
        self._setup()
        if self.cfg_train.actor_lstm:
            self.lstm_train()
        else:
            self.mlp_train()

    def get_perturbation(self, *, obs, h):
        obs_0 = obs.clone()
        with torch.no_grad():
            mu_dir, std_dir, self.hidden_paad = self.actor.get_mu_and_std(
                obs=obs, h=self.hidden_paad
            )
        mu_dir = mu_dir.reshape((obs.size(0),) + self.env.single_action_space.shape)
        std_dir = std_dir.reshape((obs.size(0),) + self.env.single_action_space.shape)
        adv_obs = self.paad_perturbation(
            observation=obs,
            eps=self.eps,
            mu_dir=mu_dir,
            std_dir=std_dir,
            step_size=self.step_size,
            num_iter=self.num_iter,
            h=h,
            deterministic_victim=self.config.deterministic,
        )
        assert torch.abs(adv_obs - obs_0).max().detach().item() <= self.eps + 1e-6
        return adv_obs.clone().detach()

    def load(self):
        raise NotImplementedError

    def save(self):
        path = f"{self.save_path}/{self.run_name}"
        os.makedirs(path, exist_ok=True)
        if self.cfg_train.checkpoint:
            torch.save(self.actor.state_dict(), f"{path}/policy.pt")
            torch.save(self.critic.state_dict(), f"{path}/critic.pt")
        config = self.cfg_train.to_dict()
        with open(f"{path}/config.json", "w") as f:
            json.dump(config, f, indent=2)
