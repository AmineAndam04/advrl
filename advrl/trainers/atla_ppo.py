import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import flatdim
from advrl.envs.mujoco import make_env_mujoco
from advrl.envs.vec_env import VecEnv
from advrl.utils.misc import set_scheduler
from advrl.agents import NETWORK_REGISTRY
from ml_collections import ConfigDict
from .base import Base_MLP_Trainer


class ATLA_PPO(Base_MLP_Trainer):
    def __init__(self, config, workdir="runs", command=""):
        super().__init__(config=config, workdir=workdir, command=command)
        self._setup()

    def _setup(self):
        super()._setup()
        #! START: adv
        self.adv_env = VecEnv(
            [
                make_env_mujoco(
                    env_id=self.config.env_id,
                    seed=self.config.seed + i,
                    gamma=self.config.atla.gamma,
                    norm_obs=self.config.norm_obs,
                    clip_obs=self.config.clip_obs,
                    range_obs=self.config.range_obs,
                    norm_reward=self.config.norm_reward,
                    clip_reward=self.config.clip_reward,
                    range_reward=self.config.range_reward,
                    update_stats=False,
                )
                for i in range(self.config.atla.num_envs)
            ]
        )
        optimizer = getattr(optim, self.config.optim)
        self.adv_actor = NETWORK_REGISTRY[self.config.atla.actor](
            in_dim=flatdim(self.env.single_observation_space),
            out_dim=flatdim(self.env.single_observation_space),
            net_arch=ConfigDict(
                {
                    "pi_net": self.config.atla.pi_net,
                    "act_fun": self.config.atla.pi_act_fun,
                    "ortho_init": self.config.atla.ortho_init,
                    "orth_std": self.config.atla.orth_std,
                    "log_std_init": self.config.atla.log_std_init,
                },
            ),
            squash=self.config.atla.squash,
        ).to(self.device)
        self.adv_critic = NETWORK_REGISTRY[self.config.atla.critic](
            in_dim=flatdim(self.env.single_observation_space),
            net_arch=ConfigDict(
                {
                    "vf_net": self.config.atla.vf_net,
                    "act_fun": self.config.atla.vf_act_fun,
                    "ortho_init": self.config.atla.ortho_init,
                    "orth_std": self.config.atla.orth_std,
                }
            ),
        ).to(self.device)
        self.adv_actor_optimizer = optimizer(
            self.adv_actor.parameters(), lr=self.config.atla.lr_actor
        )
        self.adv_critic_optimizer = optimizer(
            self.adv_critic.parameters(), lr=self.config.atla.lr_critic
        )

        self.eps_schedule = set_scheduler(config=self.config.atla)
        #! END: adv

    def _reset_rollout(self, is_adv=False):
        if is_adv:
            config = self.config.atla
            action_space = self.env.single_observation_space.shape
        else:
            config = self.config
            action_space = self.env.single_action_space.shape
        obs = torch.zeros(
            (config.n_steps, config.num_envs) + self.env.single_observation_space.shape
        ).to(self.device)
        actions = torch.zeros((config.n_steps, config.num_envs) + action_space).to(self.device)
        logprobs = torch.zeros((config.n_steps, config.num_envs)).to(self.device)
        rewards = torch.zeros((config.n_steps, config.num_envs)).to(self.device)
        dones = torch.zeros((config.n_steps, config.num_envs)).to(self.device)
        values = torch.zeros((config.n_steps, config.num_envs)).to(self.device)
        return obs, actions, logprobs, rewards, dones, values

    def collect_nsteps(
        self,
    ):
        if self.current_step == 0:
            observation, _ = self.env.reset()
            observation = torch.as_tensor(observation).to(self.device)
        else:
            observation = self.last_obs
        obs, actions, logprobs, rewards, dones, values = self._reset_rollout()
        eps = self.eps_schedule(current_t=self.current_step / self.config.total_timesteps)
        for step in range(0, self.config.n_steps):
            with torch.no_grad():
                #! START: adv
                adv_action, *_ = self.adv_actor.get_action(
                    obs=observation, deterministic=self.config.deterministic
                )
                if self.config.atla.squash:
                    adv_action = eps * adv_action
                adv_obs = observation + adv_action
                observation = torch.clamp(adv_obs, min=observation - eps, max=observation + eps)
                observation = torch.clamp(observation, min=self.low, max=self.high)
                #! END: adv
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
                                "rollout/eps": eps,
                            },
                            step=self.current_step,
                        )
                        self.ep_rewards = []
                        self.ep_lengths = []
                    if (
                        self.config.eval_every > 0
                        and self.num_episodes % self.config.eval_every == 0
                    ):
                        self.eval_flag = True

        self.last_obs = observation
        with torch.no_grad():
            last_value, _ = self.critic(self.last_obs)
        returns, advantages = self.compute_advantages_and_returns(
            values, last_value, rewards, dones
        )
        b_obs = obs.reshape((-1,) + self.env.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + self.env.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values

    def _sync_adv_env(self):
        for i in range(self.config.atla.num_envs):
            if self.config.norm_obs:
                self.adv_env.envs[i].set_wrapper_attr(
                    "obs_rms",
                    copy.deepcopy(
                        self.env.envs[i % self.config.num_envs].get_wrapper_attr("obs_rms")
                    ),
                )
            if self.config.norm_reward:
                self.adv_env.envs[i].set_wrapper_attr(
                    "return_rms",
                    copy.deepcopy(
                        self.env.envs[i % self.config.num_envs].get_wrapper_attr("return_rms")
                    ),
                )

    def collect_adv_nsteps(
        self,
    ):
        self._sync_adv_env()
        if self.adv_current_step == 0:
            observation, _ = self.adv_env.reset()
            observation = torch.as_tensor(observation).to(self.device)
        else:
            observation = self.adv_last_obs
        obs, actions, logprobs, rewards, dones, values = self._reset_rollout(is_adv=True)
        if self.config.atla.adv_scheduled:
            eps = self.eps_schedule(current_t=self.current_step / self.config.total_timesteps)
        else:
            eps = self.config.atla.eps
        for step in range(0, self.config.atla.n_steps):
            with torch.no_grad():
                action, logprob, *_ = self.adv_actor(obs=observation)
                val, _ = self.adv_critic(obs=observation)
                if self.config.atla.squash:
                    adv_action = eps * action
                else:
                    adv_action = action
                adv_obs = observation + adv_action
                adv_obs = torch.clamp(adv_obs, min=observation - eps, max=observation + eps)
                adv_obs = torch.clamp(adv_obs, min=self.low, max=self.high)
                agent_action, *_ = self.actor.get_action(
                    obs=adv_obs, deterministic=self.config.deterministic
                )
            next_obs, reward, terminations, truncations, infos = self.adv_env.step(
                agent_action.cpu().numpy()
            )
            done = np.logical_or(terminations, truncations)
            done = torch.as_tensor(done).to(self.device)
            obs[step] = observation
            rewards[step] = -torch.as_tensor(reward).to(self.device).view(-1)
            dones[step] = done
            actions[step] = action
            values[step] = val.flatten()
            logprobs[step] = logprob
            observation = torch.as_tensor(next_obs).to(self.device)
            self.adv_current_step += self.config.atla.num_envs
            for info in infos:
                if "episode" in info:
                    ep_rd = info["episode"]["r"]
                    ep_len = info["episode"]["l"]
                    self.adv_ep_rewards.append(ep_rd)
                    self.adv_ep_lengths.append(ep_len)
                    if len(self.adv_ep_rewards) == self.config.log_every:
                        self.logger.log_metrics(
                            metrics={
                                "rollout/adv_ep_reward": np.mean(self.adv_ep_rewards),
                                "rollout/adv_ep_length": np.mean(self.adv_ep_lengths),
                                "rollout/adv_eps": eps,
                            },
                            step=self.adv_current_step,
                        )
                        self.adv_ep_rewards = []
                        self.adv_ep_lengths = []
        self.adv_last_obs = observation
        with torch.no_grad():
            last_value, _ = self.adv_critic(self.adv_last_obs)
        returns, advantages = self.compute_advantages_and_returns(
            values, last_value, rewards, dones, config=self.config.atla
        )
        b_obs = obs.reshape((-1,) + self.env.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + self.env.single_observation_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values

    def train(self):
        self.adv_current_step = 0
        self.adv_ep_rewards, self.adv_ep_lengths = [], []
        start_adv = False  # (1) avoid training the adversary at (t=0) (2) avoid  the adversary being the last thing we train
        while self.current_step < self.config.total_timesteps:
            if start_adv:
                self.update_actor_and_critic(is_adv=True)
            self.update_actor_and_critic(is_adv=False)
            start_adv = True
            if self.config.num_eval_envs > 0 and self.eval_flag:
                self.evaluate()
                self.eval_flag = False
        self.evaluate()
        if self.config.checkpoint:
            self.save()

    def update_actor_and_critic(
        self,
        is_adv=False,
    ):
        if is_adv:
            actor = self.adv_actor
            actor_optimizer = self.adv_actor_optimizer
            critic = self.adv_critic
            critic_optimizer = self.adv_critic_optimizer
            config = self.config.atla
            collect_nsteps = self.collect_adv_nsteps
            prefix = "adv_train"
            current_step = self.adv_current_step
        else:
            actor = self.actor
            actor_optimizer = self.actor_optimizer
            critic = self.critic
            critic_optimizer = self.critic_optimizer
            config = self.config
            collect_nsteps = self.collect_nsteps
            prefix = "train"
            current_step = self.current_step
            if self.config.reg_coef > 0:
                eps = self.eps_schedule(current_t=self.current_step / self.config.total_timesteps)
                rb_losses = []
        # Collect env steps and compute advantage
        b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = collect_nsteps()
        # Normalize
        if config.normalize_advantage:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        if config.normalize_return:
            b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-8)
        if config.anneal_lr:
            frac = 1.0 - (current_step - 1.0) / self.config.total_timesteps
            lr_actor_now = max(config.min_lr, frac * config.lr_actor)
            lr_critic_now = max(config.min_lr, frac * config.lr_critic)
            actor_optimizer.param_groups[0]["lr"] = lr_actor_now
            critic_optimizer.param_groups[0]["lr"] = lr_critic_now
        b_inds = np.arange(config.num_envs * config.n_steps)
        v_losses, pg_losses, entropy_losses, approx_kls, clipfracs = [], [], [], [], []
        for _ in range(config.n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_inds), config.batch_size):
                end = start + config.batch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, new_mean, *_ = actor(
                    obs=b_obs[mb_inds], action=b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                mb_advantages = b_advantages[mb_inds]
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - config.clip_range, 1 + config.clip_range
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                pg_losses.append(pg_loss.item())
                entropy_loss = entropy.mean()
                entropy_losses.append(entropy_loss.item())
                loss = pg_loss - config.ent_coef * entropy_loss
                # SA-reg
                if not is_adv and self.config.reg_coef > 0 and eps > 0:
                    if self.config.solver == "sgld":
                        rb_loss = self.sgld_solver(
                            obs=b_obs[mb_inds], eps=eps, mu_0=new_mean.detach()
                        )
                    elif self.config.solver == "pi":
                        rb_loss = self.poweriter_solver(
                            obs=b_obs[mb_inds], eps=eps, mu_0=new_mean.detach()
                        )
                    rb_losses.append(rb_loss.item())
                    loss += self.config.reg_coef * rb_loss
                actor_optimizer.zero_grad()
                loss.backward()
                if config.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
                actor_optimizer.step()
                # Value loss
                newvalue, _ = critic(obs=b_obs[mb_inds])
                newvalue = newvalue.view(-1)
                if config.clip_range_vf > 0:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -config.clip_range_vf,
                        config.clip_range_vf,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                v_losses.append(v_loss.item())
                critic_optimizer.zero_grad()
                v_loss.backward()
                if config.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
                critic_optimizer.step()
                # kl
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config.clip_range).float().mean().item()]
                    approx_kls.append(approx_kl.item())
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        # Training logs
        training_metrics = {
            f"{prefix}/value_loss": np.mean(v_losses),
            f"{prefix}/policy_loss": np.mean(pg_losses),
            f"{prefix}/entropy": np.mean(entropy_losses),
            f"{prefix}/approx_kl": np.mean(approx_kls),
            f"{prefix}/clipfrac": np.mean(clipfracs),
            f"{prefix}/explained_variance": explained_var,
        }
        if config.anneal_lr:
            training_metrics.update(
                {f"{prefix}/lr_actor": lr_actor_now, f"{prefix}/lr_critic": lr_critic_now}
            )
        if not is_adv and self.config.reg_coef > 0 and eps > 0:
            training_metrics.update({f"{prefix}/rb_loss": np.mean(rb_losses)})
        self.logger.log_metrics(metrics=training_metrics, step=current_step)

    def poweriter_solver(self, obs, eps, mu_0):
        log_sigma = self.actor.logstd_head.expand_as(mu_0).clone().detach()
        sigma = torch.exp(log_sigma)
        delta = torch.zeros_like(obs).normal_()
        for _ in range(self.config.pi.num_iter):
            delta = delta.clone().detach().requires_grad_(True)
            mu, *_ = self.actor.get_mu_and_std(obs=obs + self.config.pi.xi * delta)
            kl = (((mu - mu_0) / sigma) ** 2).sum(dim=-1)
            loss = kl
            loss = loss.sum()
            loss.backward()
            delta = eps * delta.grad.sign()
        mu, *_ = self.actor.get_mu_and_std(
            obs=torch.clamp(obs + delta.detach().clone(), min=self.low, max=self.high)
        )
        kl = (((mu - mu_0) / sigma) ** 2).sum(dim=-1)
        return kl.mean()

    def sgld_solver(self, obs, eps, mu_0):
        obs_0 = obs.clone()
        log_sigma = self.actor.logstd_head.expand_as(mu_0).clone().detach()
        sigma = torch.exp(log_sigma)
        step_size = self.config.sgld.step_size
        if step_size < 0:
            step_size = eps / self.config.sgld.num_iter
        sgld_factor = np.sqrt(2 / (self.config.sgld.beta * step_size))
        for k in range(self.config.sgld.num_iter):
            obs = obs.clone().detach().requires_grad_(True)
            mu, *_ = self.actor.get_mu_and_std(obs=obs)
            kl = (((mu - mu_0) / sigma) ** 2).sum(dim=-1)
            loss = -kl
            loss = loss.sum()
            loss.backward()
            grad = obs.grad
            if k == 0:
                g = grad + torch.randn_like(obs) * sgld_factor
            else:
                g = grad + torch.randn_like(obs) * (sgld_factor / (k + 2))
            obs = obs - step_size * g.sign()
            obs = torch.clamp(obs, min=obs_0 - eps, max=obs_0 + eps)
            obs = torch.clamp(obs, min=self.low, max=self.high)
        mu, *_ = self.actor.get_mu_and_std(obs=obs.clone().detach())
        kl = (((mu - mu_0) / sigma) ** 2).sum(dim=-1)
        return kl.mean()

    def save(self):
        super().save()
        torch.save(self.adv_actor.state_dict(), f"{self.path}/adv_policy.pt")
        torch.save(self.adv_critic.state_dict(), f"{self.path}/adv_critic.pt")
