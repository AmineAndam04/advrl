import numpy as np
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from gymnasium.spaces import flatdim

from advrl.agents import RELAXED_NETWORK_REGISTRY
from advrl.utils.misc import set_scheduler

from .base import Base_MLP_Trainer


class SA_PPO(Base_MLP_Trainer):
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
        #! END: adv

    def train(self):
        while self.current_step < self.config.total_timesteps:
            # Collect env steps and compute advantage
            b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, *_ = self.collect_nsteps()
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
            if self.config.solver == "relax":
                relax_beta = self.relax_beta_schedule(
                    current_t=self.current_step / self.config.total_timesteps
                )
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
                    if eps > 0 and self.config.rb_coef > 0:
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
                        loss += self.config.rb_coef * rb_loss
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
            if eps > 0 and self.config.rb_coef > 0:
                training_metrics.update({"train/rb_loss": np.mean(rb_losses)})
                if self.config.solver == "relax":
                    training_metrics.update({"train/beta": relax_beta})
            self.logger.log_metrics(metrics=training_metrics, step=self.current_step)
            # Eval
            if self.config.num_eval_envs > 0 and self.eval_flag:
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
