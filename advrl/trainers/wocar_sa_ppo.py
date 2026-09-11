import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from gymnasium.spaces import flatdim
from ml_collections import ConfigDict

from advrl.agents import NETWORK_REGISTRY, RELAXED_NETWORK_REGISTRY
from advrl.utils.misc import set_scheduler

from .base import Base_MLP_Trainer


class WocaR_SA_PPO(Base_MLP_Trainer):
    def __init__(self, config, workdir="runs", command=""):
        super().__init__(config=config, workdir=workdir, command=command)
        self._setup()

    def _setup(self):
        super()._setup()
        optimizer = getattr(optim, self.config.optim)
        #! START: adv
        self.wc_critic = NETWORK_REGISTRY[self.config.wc_critic](
            in_dim=flatdim(self.env.single_observation_space) + flatdim(self.env.single_action_space),
            net_arch=ConfigDict(
                {
                    "q_net": self.config.wc_net,
                    "act_fun": self.config.wc_act_fun,
                    "ortho_init": self.config.ortho_init,
                    "orth_std": self.config.orth_std,
                }
            ),
        ).to(self.device)
        self.wc_target_critic = copy.deepcopy(self.wc_critic)
        self.wc_critic_optimizer = optimizer(self.wc_critic.parameters(), lr=self.config.lr_wc_critic)
        self.relaxed_actor = BoundedModule(
            model=RELAXED_NETWORK_REGISTRY[self.config.actor](mean_head=self.actor.mean_head),
            global_input=torch.randn(1, flatdim(self.env.single_observation_space)),
            device=self.device,
        )
        self.relax_beta_schedule = set_scheduler(config=self.config.relax, var_to_scheduler="beta")
        self.eps_schedule = set_scheduler(config=self.config)
        #! END: adv

    def update_wc_critic(self, b_obs, b_actions, b_rewards, b_dones):
        b_obs, b_next_obs, b_actions, b_rewards, b_dones = self.prepare_batch_for_qnets(
            b_obs, b_actions, b_rewards, b_dones
        )
        if self.config.normalize_rewards:
            b_rewards = (b_rewards - b_rewards.mean()) / (b_rewards.std() + 1e-8)
        b_inds = np.arange(self.config.num_envs * (self.config.n_steps - 1))
        eps = self.eps_schedule(current_t=self.current_step / self.config.total_timesteps)
        relax_beta = self.relax_beta_schedule(current_t=self.current_step / self.config.total_timesteps)
        td_losses = []
        for _ in range(self.config.wc_n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_inds), self.config.batch_size):
                end = start + self.config.batch_size
                mb_inds = b_inds[start:end]
                with torch.no_grad():
                    lb, ub = self.compute_bounds(obs=b_next_obs[mb_inds], eps=eps, beta=relax_beta)
                min_next_action, _ = self.constrained_solver(lb=lb, ub=ub, b_obs=b_next_obs[mb_inds])
                with torch.no_grad():
                    next_qvals, _ = self.wc_target_critic(
                        obs_action=torch.cat([b_next_obs[mb_inds], min_next_action], dim=-1)
                    )
                    td_target = (
                        b_rewards[mb_inds] + self.config.gamma * (1 - b_dones[mb_inds]) * next_qvals.squeeze()
                    )
                q_values, _ = self.wc_critic(
                    obs_action=torch.cat([b_obs[mb_inds], b_actions[mb_inds]], dim=-1)
                )
                q_values = q_values.squeeze()
                td_loss = nn.functional.mse_loss(q_values, td_target)
                td_losses.append(td_loss.item())
                self.wc_critic_optimizer.zero_grad()
                td_loss.backward()
                if self.config.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.wc_critic.parameters(), self.config.max_grad_norm)
                self.wc_critic_optimizer.step()
        self.logger.log_metrics(
            metrics={"train/wc_qvalue_loss": np.mean(td_losses)},
            step=self.current_step,
        )
        for target_param, param in zip(self.wc_target_critic.parameters(), self.wc_critic.parameters()):
            target_param.data.copy_(
                self.config.wc_polyak * param.data + (1.0 - self.config.wc_polyak) * target_param.data
            )

    def update_actor_and_critic(self, b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values):
        b_inds = np.arange(self.config.num_envs * self.config.n_steps)
        pg_losses, entropy_losses, approx_kls, clipfracs, reg_losses, v_losses = [], [], [], [], [], []
        eps = self.eps_schedule(current_t=self.current_step / self.config.total_timesteps)
        relax_beta = self.relax_beta_schedule(current_t=self.current_step / self.config.total_timesteps)
        if eps > 0 and self.config.wc_coef > 0:
            ## Q values
            with torch.no_grad():
                q_values, _ = self.wc_critic(torch.cat([b_obs, b_actions], dim=-1))
            b_wc_advantages = q_values.squeeze()
            if self.config.normalize_advantage:
                b_wc_advantages = (b_wc_advantages - b_wc_advantages.mean()) / (b_wc_advantages.std() + 1e-8)

        for _ in range(self.config.n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_inds), self.config.batch_size):
                end = start + self.config.batch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, new_mean, *_ = self.actor(
                    obs=b_obs[mb_inds], action=b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                adv_advantages = b_advantages[mb_inds]
                #! STAT: adv
                if eps > 0 and self.config.wc_coef > 0:
                    adv_advantages += self.config.wc_coef * b_wc_advantages[mb_inds]
                #! END: adv
                # Policy loss
                pg_loss1 = -adv_advantages * ratio
                pg_loss2 = -adv_advantages * torch.clamp(
                    ratio, 1 - self.config.clip_range, 1 + self.config.clip_range
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                pg_losses.append(pg_loss.item())
                entropy_loss = entropy.mean()
                entropy_losses.append(entropy_loss.item())
                loss = pg_loss - self.config.ent_coef * entropy_loss
                #! STAT: adv
                # reg loss
                if eps > 0 and self.config.reg_coef > 0:
                    kl = self.relax_solver(obs=b_obs[mb_inds], eps=eps, beta=relax_beta, mu_0=new_mean)
                    reg_loss = kl.mean()
                    reg_losses.append(reg_loss.item())
                    loss += self.config.reg_coef * reg_loss
                #! END: adv
                self.actor_optimizer.zero_grad()
                loss.backward()
                if self.config.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()
                # update critic
                v_loss = self.update_critic(
                    mb_obs=b_obs[mb_inds], mb_returns=b_returns[mb_inds], mb_values=b_values[mb_inds]
                )
                v_losses.append(v_loss)
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.config.clip_range).float().mean().item()]
                    approx_kls.append(approx_kl.item())
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        # Log
        training_metrics = {
            "train/policy_loss": np.mean(pg_losses),
            "train/entropy": np.mean(entropy_losses),
            "train/approx_kl": np.mean(approx_kls),
            "train/clipfrac": np.mean(clipfracs),
            "train/value_loss": np.mean(v_losses),
            "train/explained_variance": explained_var,
            "train/eps": eps,
        }
        if eps > 0 and self.config.reg_coef > 0:
            training_metrics.update({"train/reg_loss": np.mean(reg_losses)})
            training_metrics.update({"train/beta": relax_beta})
        self.logger.log_metrics(metrics=training_metrics, step=self.current_step)

    def train(self):
        while self.current_step < self.config.total_timesteps:
            b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_rewards, b_dones = (
                self.collect_nsteps()
            )
            if self.config.normalize_advantage:
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            if self.config.normalize_return:
                b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-8)
            if self.config.anneal_lr:
                # for actor and critic
                self.update_lr()
                frac = 1.0 - (self.current_step - 1.0) / self.config.total_timesteps
                lr_wc_now = max(self.config.min_lr, frac * self.config.lr_wc)
                self.wc_critic_optimizer.param_groups[0]["lr"] = lr_wc_now
                self.logger.log_metrics(metrics={"train/lr_wc": lr_wc_now}, step=self.current_step)
            eps = self.eps_schedule(current_t=self.current_step / self.config.total_timesteps)
            # update the worst case critic
            if eps > 0:
                self.update_wc_critic(b_obs=b_obs, b_actions=b_actions, b_rewards=b_rewards, b_dones=b_dones)
            # update the actor and critic
            self.update_actor_and_critic(b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values)
            if self.eval_flag:
                self.evaluate()
                self.eval_flag = False
        self.evaluate()
        if self.config.checkpoint:
            self.save()

    def prepare_batch_for_qnets(self, b_obs, b_actions, b_rewards, b_dones):
        obs = b_obs.clone().reshape(
            (self.config.n_steps, self.config.num_envs) + self.env.single_observation_space.shape
        )
        actions = b_actions.clone().reshape(
            (self.config.n_steps, self.config.num_envs) + self.env.single_action_space.shape
        )
        rewards = b_rewards.clone().reshape((self.config.n_steps, self.config.num_envs))
        dones = b_dones.clone().reshape((self.config.n_steps, self.config.num_envs))
        b_obs = obs[:-1].reshape((-1,) + self.env.single_observation_space.shape)
        b_next_obs = obs[1:].reshape((-1,) + self.env.single_observation_space.shape)
        b_actions = actions[:-1].reshape((-1,) + self.env.single_action_space.shape)
        b_rewards = rewards[:-1].reshape(-1)
        b_dones = dones[:-1].reshape(-1)
        return b_obs, b_next_obs, b_actions, b_rewards, b_dones

    def compute_bounds(self, obs, eps, beta):
        ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
        bounded_obs = BoundedTensor(obs, ptb)
        # IBP bounds
        if beta > 0:
            ibp_lb, ibp_ub = self.relaxed_actor.compute_bounds(x=(bounded_obs,), method="ibp")
        else:
            ibp_lb, ibp_ub = 0, 0
        # CROWN-IBP
        if (1 - beta) > 0:
            cr_x = None if beta > 0 else (bounded_obs,)
            cr_lb, cr_ub = self.relaxed_actor.compute_bounds(x=cr_x, method="backward")
        else:
            cr_lb, cr_ub = 0, 0
        lb = beta * ibp_lb + (1 - beta) * cr_lb
        ub = beta * ibp_ub + (1 - beta) * cr_ub
        return lb, ub

    def relax_solver(self, obs, eps, beta, mu_0):
        log_sigma = self.actor.logstd_head.expand_as(mu_0).clone()
        sigma = torch.exp(log_sigma)
        lb, ub = self.compute_bounds(obs=obs, eps=eps, beta=beta)
        diff = torch.max(torch.abs(lb - mu_0), torch.abs(ub - mu_0))
        kl = (((diff) / sigma) ** 2).sum(dim=-1)
        return kl

    def constrained_solver(self, lb, ub, b_obs, get_min_q=False):
        action = ((lb + ub) / 2.0).detach().clone().requires_grad_(True)
        wc_target_lr = (ub - lb) / self.config.wc_target_iter
        for _ in range(self.config.wc_target_iter):
            action = action.clone().detach().requires_grad_(True)
            obs_action = torch.cat([b_obs, action], dim=-1)
            qvals, *_ = self.wc_critic(obs_action=obs_action)
            qvals = qvals.sum()
            qvals.backward()
            action = action - wc_target_lr * action.grad.data
            action = torch.clamp(action, lb, ub)
        if get_min_q:
            with torch.no_grad():
                obs_action = torch.cat([b_obs, action.clone().detach()], dim=-1)
                qvals, *_ = self.wc_critic(obs_action=obs_action)
        return action.detach(), qvals.detach()

    def save(self):
        super().save()
        torch.save(self.wc_critic.state_dict(), f"{self.path}/wc_critic.pt")
