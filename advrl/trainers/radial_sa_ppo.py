import numpy as np
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from gymnasium.spaces import flatdim

from advrl.agents import RELAXED_NETWORK_REGISTRY
from advrl.utils.misc import set_scheduler

from .base import Base_MLP_Trainer


class Radial_SA_PPO(Base_MLP_Trainer):
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
        #! END: adv

    def train(self):
        while self.current_step < self.config.total_timesteps:
            b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, *_ = self.collect_nsteps()
            # Normalize
            if self.config.normalize_advantage:
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            if self.config.normalize_return:
                b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-8)
            if self.config.anneal_lr:
                self.update_lr()
            b_inds = np.arange(self.config.num_envs * self.config.n_steps)
            v_losses, pg_losses, radial_losses, sa_reg_losses, entropy_losses, approx_kls, clipfracs = (
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
                    if eps > 0 and self.config.radial_coef > 0 and self.config.reg_coef > 0:
                        lb_logprob, ub_logprob, sa_reg = self.relax_solver(
                            obs=b_obs[mb_inds],
                            eps=eps,
                            action=b_actions[mb_inds],
                            beta=relax_beta,
                            mu_0=new_mean,
                        )
                        rb_logprob = torch.where(mb_advantages >= 0, lb_logprob, ub_logprob)
                        rb_logratio = rb_logprob - b_logprobs[mb_inds]
                        rb_ratio = rb_logratio.exp()
                        rb_loss1 = -mb_advantages * rb_ratio
                        rb_loss2 = -mb_advantages * torch.clamp(
                            rb_ratio, 1 - self.config.clip_range, 1 + self.config.clip_range
                        )
                        radial_loss = torch.max(rb_loss1, rb_loss2).mean()
                        radial_losses.append(radial_loss.item())
                        sa_reg_losses.append(sa_reg.item())
                        loss += self.config.radial_coef * radial_loss + self.config.reg_coef * sa_reg
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
            # Training logs
            training_metrics = {
                "train/value_loss": np.mean(v_losses),
                "train/policy_loss": np.mean(pg_losses),
                "train/entropy": np.mean(entropy_losses),
                "train/approx_kl": np.mean(approx_kls),
                "train/clipfrac": np.mean(clipfracs),
                "train/explained_variance": explained_var,
                "train/eps": eps,
                "train/beta": relax_beta,
            }
            if eps > 0 and self.config.radial_coef > 0 and self.config.reg_coef > 0:
                training_metrics.update(
                    {"train/radial_loss": np.mean(radial_losses), "train/reg_loss": np.mean(sa_reg_losses)}
                )
            self.logger.log_metrics(metrics=training_metrics, step=self.current_step)
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
        # Radial
        mu_lb = beta * ibp_lb + (1 - beta) * cr_lb
        mu_ub = beta * ibp_ub + (1 - beta) * cr_ub
        d_lb = (((action - torch.clamp(action, mu_lb, mu_ub)) / sigma) ** 2).sum(dim=-1)
        d_ub = ((torch.max(torch.abs(mu_lb - action), torch.abs(mu_ub - action)) / sigma) ** 2).sum(dim=-1)
        log_denom = 0.5 * action.shape[-1] * np.log(2 * np.pi) + torch.log(sigma).sum(dim=-1)
        lb_log_pi = -0.5 * d_ub - log_denom
        ub_log_pi = -0.5 * d_lb - log_denom
        # SA-Reg
        lb = beta * ibp_lb + (1 - beta) * cr_lb
        ub = beta * ibp_ub + (1 - beta) * cr_ub
        diff = torch.max(torch.abs(lb - mu_0), torch.abs(ub - mu_0))
        kl = (((diff) / sigma) ** 2).sum(dim=-1)
        return lb_log_pi, ub_log_pi, kl.mean()
