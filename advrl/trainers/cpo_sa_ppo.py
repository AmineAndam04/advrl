from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from gymnasium.spaces import flatdim

from advrl.agents import RELAXED_NETWORK_REGISTRY
from advrl.utils.misc import grad_norm, set_scheduler

from .base import Base_MLP_Trainer


class CPO_SA_PPO(Base_MLP_Trainer):
    def __init__(self, config, workdir="runs", command=""):
        super().__init__(config=config, workdir=workdir, command=command)
        self._setup()

    def _setup(self):
        super()._setup()
        #! START: adv
        if self.config.solver == "relax":
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
        vals = []  #! START adv
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
            for i, info in enumerate(infos):
                if "episode" in info:
                    self.num_episodes += 1
                    ep_rd = info["episode"]["r"]
                    ep_len = info["episode"]["l"]
                    #! START adv
                    with torch.no_grad():
                        val, _ = self.critic(obs=observation[i])
                        vals.append(val.item())
                    #! END adv
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

        #! START adv
        self.J_pi_k = np.mean(vals + values[0].tolist())
        self.alpha = 2 * min(rewards.abs().max().item(), 1) / (1 - self.config.gamma) ** 2
        #! END adv
        with torch.no_grad():
            last_value, _ = self.critic(observation)
        returns, advantages = self.compute_advantages_and_returns(values, last_value, rewards, dones)
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
            v_losses, pg_losses, rb_losses, entropy_losses, approx_kls, clipfracs = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            eps = self.eps_schedule(current_t=self.current_step / self.config.total_timesteps)

            #! START adv
            if eps > 0:
                if self.config.solver == "relax":
                    relax_beta = self.relax_beta_schedule(
                        current_t=self.current_step / self.config.total_timesteps
                    )
                # Update lagrangian multiplier
                self.update_lagrangian_multiplier(
                    b_obs=b_obs, b_actions=b_actions, eps=eps, relax_beta=relax_beta
                )
            if self.config.balance_grad:
                betas = []
            #! END adv
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
                        if self.config.solver == "relax":
                            rb_loss = self.relax_solver(
                                obs=b_obs[mb_inds], eps=eps, beta=relax_beta, mu_0=new_mean
                            )
                        elif self.config.solver == "sgld":
                            rb_loss = self.sgld_solver(obs=b_obs[mb_inds], eps=eps, mu_0=new_mean.detach())
                        elif self.config.solver == "pi":
                            rb_loss = self.poweriter_solver(
                                obs=b_obs[mb_inds], eps=eps, mu_0=new_mean.detach()
                            )
                        rb_losses.append(rb_loss.item())
                        if self.config.balance_grad:
                            beta = self.beta_to_balance_grad(pg_loss=pg_loss, reg_loss=rb_loss)
                            betas.append(beta)
                        else:
                            beta = 1
                        loss += beta * self.lagrangian_multiplier.item() * rb_loss
                        loss /= 1 + self.lagrangian_multiplier
                    #! END: adv
                    self.actor_optimizer.zero_grad()
                    loss.backward()
                    if self.config.max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                    self.actor_optimizer.step()
                    # Value loss
                    v_loss = self.update_critic(
                        mb_obs=b_obs[mb_inds], mb_returns=b_returns[mb_inds], mb_values=b_values[mb_inds]
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
            # Training logs
            training_metrics = {
                "train/value_loss": np.mean(v_losses),
                "train/policy_loss": np.mean(pg_losses),
                "train/entropy": np.mean(entropy_losses),
                "train/approx_kl": np.mean(approx_kls),
                "train/clipfrac": np.mean(clipfracs),
                "train/explained_variance": explained_var,
                "train/eps": eps,
            }

            if eps > 0 and self.lagrangian_multiplier.item() > 0:
                training_metrics.update({"train/rb_loss": np.mean(rb_losses)})
                if self.config.solver == "relax":
                    training_metrics.update({"train/relax_beta": relax_beta})
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

    def relax_solver(self, obs, eps, beta, mu_0):
        log_sigma = self.actor.logstd_head.expand_as(mu_0).clone()
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
        lb = beta * ibp_lb + (1 - beta) * cr_lb
        ub = beta * ibp_ub + (1 - beta) * cr_ub
        diff = torch.max(torch.abs(lb - mu_0), torch.abs(ub - mu_0))
        kl = (((diff) / sigma) ** 2).sum(dim=-1)
        return kl.mean()

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

    def update_lagrangian_multiplier(self, b_obs, b_actions, eps, relax_beta):
        with torch.no_grad():
            _, _, _, mu_0, *_ = self.actor(obs=b_obs, action=b_actions)
            if self.config.solver == "relax":
                rb_loss = self.relax_solver(obs=b_obs, eps=eps, beta=relax_beta, mu_0=mu_0)
            elif self.config.solver == "sgld":
                rb_loss = self.sgld_solver(obs=b_obs, eps=eps, mu_0=mu_0)
            elif self.config.solver == "pi":
                rb_loss = self.poweriter_solver(obs=b_obs, eps=eps, mu_0=mu_0)
        cost = self.alpha * ((0.5 * rb_loss) ** 0.5)
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
                "train/penality_loss": penality_loss.item(),
                "train/limit": limit.item(),
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
