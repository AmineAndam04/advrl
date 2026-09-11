from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from gymnasium.spaces import flatdim
from torch.distributions.normal import Normal

from advrl.agents import RELAXED_NETWORK_REGISTRY
from advrl.utils.misc import grad_norm, set_scheduler

from .base import Base_MLP_Trainer


class CPO_Radial_PPO(Base_MLP_Trainer):
    def __init__(self, config, workdir="runs", command=""):
        super().__init__(config=config, workdir=workdir, command=command)
        self._setup()

    def _setup(self):
        super()._setup()
        #! START: adv
        self.relaxed_actor = BoundedModule(
            model=RELAXED_NETWORK_REGISTRY[self.config.actor](mean_head=self.actor.mean_head),
            global_input=torch.randn(1, flatdim(self.env.single_observation_space)),
            device=self.device,
        )
        self.relax_beta_schedule = set_scheduler(config=self.config.relax, var_to_scheduler="beta")
        self.eps_schedule = set_scheduler(config=self.config)
        # PPO-LAG
        if self.config.use_pid:
            self.lagrangian_multiplier = torch.tensor(self.config.ki)
            self.pid_i = self.config.ki
            self.delta_p = 0
            self.cost_d = 0
            self.cost_ds = deque(maxlen=self.config.max_deque)
            self.cost_ds.append(0.0)
        else:
            self.lagrangian_multiplier = torch.nn.Parameter(
                torch.as_tensor(self.config.lag_init), requires_grad=True
            )
            optimizer = getattr(optim, self.config.optim)
            self.lambda_optimizer = optimizer([self.lagrangian_multiplier], lr=self.config.lag_lr)
        #! END: adv

    def collect_nsteps(
        self,
    ):
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
        observation, _ = self.env.reset()
        observation = torch.Tensor(observation).to(self.device)
        vals = []  #! adv
        for step in range(0, self.config.n_steps):
            with torch.no_grad():
                action, logprob, *_ = self.actor(obs=observation)
                val, _ = self.critic(obs=observation)
            next_obs, reward, terminations, truncations, infos = self.env.step(action.cpu().numpy())
            done = np.logical_or(terminations, truncations)
            done = torch.Tensor(done).to(self.device)
            obs[step] = observation
            rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            dones[step] = done
            actions[step] = action
            values[step] = val.flatten()
            logprobs[step] = logprob
            observation = torch.Tensor(next_obs).to(self.device)
            self.current_step += self.config.num_envs
            for i, info in enumerate(infos):
                if "episode" in info:
                    self.num_episodes += 1
                    ep_rd = info["episode"]["r"]
                    ep_len = info["episode"]["l"]
                    #! START: adv
                    with torch.no_grad():
                        val, _ = self.critic(obs=observation[i])
                        vals.append(val.item())
                    #! END: adv
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
        with torch.no_grad():
            last_value, _ = self.critic(observation)
        returns, advantages = self.compute_advantages_and_returns(values, last_value, rewards, dones)
        #! START adv
        self.J_pi_k = np.mean(vals + values[0].tolist())
        self.alpha = (
            2
            * self.config.gamma
            * torch.quantile(advantages.abs(), 0.95).item()
            / ((1 - self.config.gamma) ** 2)
        )
        if self.config.max_alpha > 0:
            self.alpha = min(self.config.max_alpha, self.alpha)
        #! END adv
        b_obs = obs.reshape((-1,) + self.env.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + self.env.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values

    def train(self):
        while self.current_step < self.config.total_timesteps:
            # Collect env steps and compute advantage
            b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = self.collect_nsteps()
            # Normalize
            if self.config.normalize_advantage:
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            if self.config.normalize_return:
                b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-8)
            if self.config.anneal_lr:
                self.update_lr()
            b_inds = np.arange(self.config.num_envs * self.config.n_steps)
            v_losses, pg_losses, radial_losses, entropy_losses, approx_kls, clipfracs, sa_reg_losses = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            eps = self.eps_schedule(current_t=self.current_step / self.config.total_timesteps)
            relax_beta = self.relax_beta_schedule(current_t=self.current_step / self.config.total_timesteps)
            if eps > 0:
                # Update lagrangian multiplier
                self.update_lagrangian_multiplier(
                    b_obs=b_obs,
                    b_actions=b_actions,
                    b_logprobs=b_logprobs,
                    b_advantages=b_advantages,
                    eps=eps,
                    relax_beta=relax_beta,
                )
                if self.config.balance_grad:
                    betas = []
            for _ in range(self.config.n_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, len(b_inds), self.config.batch_size):
                    end = start + self.config.batch_size
                    mb_inds = b_inds[start:end]
                    _, newlogprob, entropy, mu_new, *_ = self.actor(
                        obs=b_obs[mb_inds], action=b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    mb_advantages = b_advantages[mb_inds]
                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.config.clip_range, 1 + self.config.clip_range
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    pg_losses.append(pg_loss.item())
                    entropy_loss = entropy.mean()
                    entropy_losses.append(entropy_loss.item())
                    loss = pg_loss - self.config.ent_coef * entropy_loss
                    #! START: adv
                    if eps > 0 and self.lagrangian_multiplier.item() > 0:
                        logprob1, logprob2, sa_reg = self.relax_solver(
                            obs=b_obs[mb_inds],
                            eps=eps,
                            action=b_actions[mb_inds],
                            beta=relax_beta,
                            mu_0=mu_new,
                        )
                        if self.config.radial_wc:
                            rb_logprob = torch.where(mb_advantages >= 0, logprob1, logprob2)
                        else:
                            rb_logprob = logprob1
                        rb_logratio = rb_logprob - b_logprobs[mb_inds]
                        rb_ratio = rb_logratio.exp()
                        radial_loss1 = -mb_advantages * rb_ratio
                        radial_loss2 = -mb_advantages * torch.clamp(
                            rb_ratio, 1 - self.config.clip_range, 1 + self.config.clip_range
                        )
                        radial_loss = torch.max(radial_loss1, radial_loss2).mean()
                        radial_losses.append(radial_loss.item())
                        sa_reg_losses.append(sa_reg.item())
                        if self.config.reg_coef < 0:
                            rb_loss = radial_loss + (1 - self.config.gamma) * self.alpha * (
                                (0.5 * sa_reg) ** 0.5
                            )
                        else:
                            rb_loss = radial_loss + self.config.reg_coef * ((0.5 * sa_reg) ** 0.5)
                        if self.config.balance_grad:
                            beta = self.beta_to_balance_grad(pg_loss=pg_loss, reg_loss=rb_loss)
                            betas.append(beta)
                        else:
                            beta = 1
                        loss += beta * self.lagrangian_multiplier.item() * rb_loss
                        loss /= 1 + self.lagrangian_multiplier.item()
                    #! END: adv
                    self.actor_optimizer.zero_grad()
                    loss.backward()
                    if self.config.max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                    self.actor_optimizer.step()
                    # Value loss
                    v_loss = self.update_critic(
                        mb_obs=b_obs[mb_inds],
                        mb_returns=b_returns[mb_inds],
                        mb_values=b_values[mb_inds],
                    )
                    v_losses.append(v_loss)
                    # kl
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.config.clip_range).float().mean().item()]
                        approx_kls.append(approx_kl.item())
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            training_metrics = {
                "train/value_loss": np.mean(v_losses),
                "train/policy_loss": np.mean(pg_losses),
                "train/entropy": np.mean(entropy_losses),
                "train/approx_kl": np.mean(approx_kls),
                "train/clipfrac": np.mean(clipfracs),
                "train/explained_variance": explained_var,
                "train/eps": eps,
                "train/relax_beta": relax_beta,
            }
            if eps > 0 and self.lagrangian_multiplier.item() > 0:
                training_metrics.update({"train/radial_loss": np.mean(radial_losses)})
                training_metrics.update({"train/sa_reg": np.mean(sa_reg_losses)})
                if self.config.balance_grad:
                    training_metrics.update({"train/grad_beta": np.mean(betas)})
            self.logger.log_metrics(metrics=training_metrics, step=self.current_step)
            # Eval
            if self.eval_flag:
                self.evaluate()
                self.eval_flag = False
        self.evaluate()
        if self.config.checkpoint:
            self.save()

    def relax_solver(self, obs, eps, action, beta, mu_0):
        log_sigma = self.actor.logstd_head.expand_as(action).clone()
        sigma = torch.exp(log_sigma)
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
        # SA-Reg
        lb = beta * ibp_lb + (1 - beta) * cr_lb
        ub = beta * ibp_ub + (1 - beta) * cr_ub
        diff = torch.max(torch.abs(lb - mu_0), torch.abs(ub - mu_0))
        kl = (((diff) / sigma) ** 2).sum(dim=-1)
        # Radial
        mu_lb = beta * ibp_lb + (1 - beta) * cr_lb
        mu_ub = beta * ibp_ub + (1 - beta) * cr_ub
        if self.config.radial_wc:
            d_lb = (((action - torch.clamp(action, mu_lb, mu_ub)) / sigma) ** 2).sum(dim=-1)
            d_ub = ((torch.max(torch.abs(mu_lb - action), torch.abs(mu_ub - action)) / sigma) ** 2).sum(
                dim=-1
            )
            log_denom = 0.5 * action.shape[-1] * np.log(2 * np.pi) + torch.log(sigma).sum(dim=-1)
            lb_log_pi = -0.5 * d_ub - log_denom
            ub_log_pi = -0.5 * d_lb - log_denom
            return lb_log_pi, ub_log_pi, 0.5 * kl.mean()
        else:
            wc_mu = torch.where(torch.abs(mu_lb - mu_0) > torch.abs(mu_ub - mu_0), mu_lb, mu_ub)
            probs = Normal(wc_mu, sigma)
            return probs.log_prob(action).sum(-1), None, 0.5 * kl.mean()

    def update_lagrangian_multiplier(self, b_obs, b_actions, b_logprobs, b_advantages, eps, relax_beta):
        with torch.no_grad():
            _, _, _, mu, *_ = self.actor(obs=b_obs, action=b_actions)
            logprob1, logprob2, sa_reg = self.relax_solver(
                obs=b_obs, eps=eps, action=b_actions, beta=relax_beta, mu_0=mu
            )
        rb_logprob = torch.where(b_advantages >= 0, logprob1, logprob2) if self.config.radial_wc else logprob1
        rb_logratio = rb_logprob - b_logprobs
        rb_ratio = rb_logratio.exp()
        radial_loss1 = b_advantages * rb_ratio
        radial_loss2 = b_advantages * torch.clamp(
            rb_ratio, 1 - self.config.clip_range, 1 + self.config.clip_range
        )
        radial_loss = torch.min(radial_loss1, radial_loss2).mean()
        cost = (-1 / (1 - self.config.gamma)) * radial_loss + self.alpha * ((0.5 * sa_reg) ** 0.5)
        limit = self.config.tolerance * np.abs(self.J_pi_k)
        if self.config.use_pid:
            self.update_by_pid(cost=cost, limit=limit)
        else:
            penality_loss = -self.lagrangian_multiplier * (cost - limit)
            self.lambda_optimizer.zero_grad()
            penality_loss.backward()
            self.lambda_optimizer.step()
        self.lagrangian_multiplier.data.clamp_(0.0, self.config.lag_max)
        self.logger.log_metrics(
            metrics={
                "train/lag": self.lagrangian_multiplier.item(),
                "train/cost": cost.item(),
            },
            step=self.current_step,
        )

    def update_by_pid(self, cost, limit):
        delta = cost - limit
        self.pid_i = max(0, self.pid_i + delta * self.config.ki)
        self.delta_p = self.config.ema * self.delta_p + (1 - self.config.ema) * delta
        self.cost_d = self.config.ema * self.cost_d + (1 - self.config.ema) * cost
        self.pid_d = max(0.0, self.cost_d - self.cost_ds[0])
        lag = self.config.kp * self.delta_p + self.pid_i + self.config.kd * self.pid_d
        self.lagrangian_multiplier = torch.clamp(lag, min=0)
        self.cost_ds.append(self.cost_d)

    def beta_to_balance_grad(self, pg_loss, reg_loss):
        reward_norm = grad_norm(loss=pg_loss, net=self.actor)
        reg_norm = grad_norm(loss=reg_loss, net=self.actor)
        beta = (reward_norm / (reg_norm + 1e-8)).detach()
        return beta
