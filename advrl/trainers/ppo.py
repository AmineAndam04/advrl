import numpy as np
import torch
import torch.nn as nn
from .base import Base_MLP_Trainer


class PPO(Base_MLP_Trainer):
    def __init__(self, config, workdir="runs", command=""):
        super().__init__(config=config, workdir=workdir, command=command)
        super()._setup()

    def train(self):
        while self.current_step < self.config.total_timesteps:
            # Collect env steps and compute advantage
            b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, *_ = (
                self.collect_nsteps()
            )
            # normalize
            if self.config.normalize_advantage:
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            if self.config.normalize_return:
                b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-8)
            if self.config.anneal_lr:
                self.update_lr()

            b_inds = np.arange(self.config.num_envs * self.config.n_steps)
            v_losses, pg_losses, entropy_losses, approx_kls, clipfracs = [], [], [], [], []

            for _ in range(self.config.n_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, len(b_inds), self.config.batch_size):
                    end = start + self.config.batch_size
                    mb_inds = b_inds[start:end]
                    _, newlogprob, entropy, *_ = self.actor(
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
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.config.clip_range).float().mean().item()
                        ]
                        approx_kls.append(approx_kl.item())

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            # Training logs
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
            # Eval
            if self.config.num_eval_envs > 0 and self.eval_flag:
                self.evaluate()
                self.eval_flag = False
        self.evaluate()
        if self.config.checkpoint:
            self.save()
