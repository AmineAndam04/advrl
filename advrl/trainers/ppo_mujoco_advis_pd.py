import os
import json
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import flatdim
from ml_collections import ConfigDict
from advrl.envs.mujoco import make_env_mujoco
from advrl.envs.vec_env import EpisodicVecEnv
from advrl.utils.misc import set_random_seed, set_device, set_scheduler
from advrl.utils.save import env_metadata
from advrl.utils.logger import Logger
from advrl.agents import NETWORK_REGISTRY, RELAXED_NETWORK_REGISTRY
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import torch.nn.functional as F


class PPO_MuJoCo_PDIS:
    def __init__(self, config, workdir="runs", command=""):
        self.config = config
        self.workdir = workdir
        self.command = command
        self.setup_()

    def setup_(self):
        """
        Prepare the environment, optimizers, seeds ...
        """
        set_random_seed(self.config.seed)
        self.env = EpisodicVecEnv(
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
            self.eval_env = EpisodicVecEnv(
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
        #! START: adv
        self.relaxed_actor = BoundedModule(
            model=RELAXED_NETWORK_REGISTRY[self.config.actor](mean_head=self.actor.mean_head),
            global_input=torch.randn(1, flatdim(self.env.single_observation_space)),
            device=self.device,
        )
        self.relax_beta_schedule = set_scheduler(config=self.config.relax, var_to_scheduler="beta")
        self.eps_schedule = set_scheduler(config=self.config)
        self.adv_loss_fun = self.get_which_adv_loss()
        #! END: adv
        # Logger
        time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_name = f"ADV-PDIS-PPO-MLP-{self.config.exp_name}__{time_token}"
        self.logger = Logger(
            mode=self.config.logger,
            wand_project=self.config.project_name,
            run_name=self.run_name,
            log_dir=f"{self.workdir}/{self.config.env_id}/advis_pd",
            config=self.config,
        )
        # Save config and command
        path = f"{self.workdir}/{self.config.env_id}/advis_pd/{self.run_name}"
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        with open(f"{path}/command.txt", "w") as f:
            f.write("python " + " ".join(self.command) + "\n")

    def collect_nsteps(
        self,
    ):
        observation, _ = self.env.reset()
        observation = torch.Tensor(observation).to(self.device)
        alive_envs = self.env.alive_envs
        episodes = [
            {
                "obs": [],
                "actions": [],
                "logprobs": [],
                "rewards": [],
                "values": [],
            }
            for _ in range(self.config.num_envs)
        ]
        while len(alive_envs) > 0:
            with torch.no_grad():
                action, logprob, *_ = self.actor(obs=observation)
                value, _ = self.critic(obs=observation)
            next_obs, reward, terminated, truncated, infos = self.env.step(action.cpu().numpy())
            reward = torch.tensor(reward).to(self.device).view(-1)
            value = value.flatten()
            for i, j in enumerate(alive_envs):
                episodes[j]["obs"].append(observation[i])
                episodes[j]["actions"].append(action[i])
                episodes[j]["logprobs"].append(logprob[i])
                episodes[j]["rewards"].append(reward[i])
                episodes[j]["values"].append(value[i])

            self.current_step += len(alive_envs)
            alive_envs = self.env.alive_envs
            done = np.logical_or(terminated, truncated)
            observation = torch.Tensor(next_obs).to(self.device)
            observation = observation[~done]
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
                        self.config.eval_every > 0
                        and self.num_episodes % self.config.eval_every == 0
                    ):
                        self.eval_flag = True
        (
            b_obs,
            b_actions,
            b_logprobs,
            b_rewards,
            b_values,
            b_mask,
        ) = self.batchify(episodes)
        b_advantages, b_returns = self.compute_advantages_and_returns(b_rewards, b_values)
        return (
            b_obs,
            b_logprobs,
            b_actions,
            b_rewards,
            b_advantages,
            b_returns,
            b_values,
            b_mask,
        )

    def compute_advantages_and_returns(self, b_rewards, b_values):
        advantages = torch.zeros_like(b_rewards).to(self.device)
        lastgaelam = 0
        for t in reversed(range(b_rewards.shape[1])):
            if t == (b_rewards.shape[1] - 1):
                next_value = 0
            else:
                next_value = b_values[:, t + 1]
            delta = b_rewards[:, t] + self.config.gamma * next_value - b_values[:, t]
            advantages[:, t] = lastgaelam = (
                delta + self.config.gamma * self.config.gae_lambda * lastgaelam
            )

        returns = advantages + b_values
        return advantages, returns

    def train(self):
        self.current_step = 0
        self.num_episodes = 0
        self.eval_flag = False
        self.ep_rewards, self.ep_lengths = [], []
        while self.current_step < self.config.total_timesteps:
            (
                b_obs,
                b_logprobs,
                b_actions,
                b_rewards,
                b_advantages,
                b_returns,
                b_values,
                b_mask,
            ) = self.collect_nsteps()
            org_b_advantage = b_advantages.clone()
            if self.config.normalize_advantage:
                b_advantages = (b_advantages - b_advantages[b_mask].mean()) / (
                    b_advantages[b_mask].std() + 1e-8
                )
            if self.config.normalize_return:
                b_returns = (b_returns - b_returns[b_mask].mean()) / (
                    b_returns[b_mask].std() + 1e-8
                )
            if self.config.anneal_lr:
                frac = 1.0 - (self.current_step - 1.0) / self.config.total_timesteps
                lr_actor_now = max(self.config.min_lr, frac * self.config.lr_actor)
                lr_critic_now = max(self.config.min_lr, frac * self.config.lr_critic)
                self.actor_optimizer.param_groups[0]["lr"] = lr_actor_now
                self.critic_optimizer.param_groups[0]["lr"] = lr_critic_now

            (
                v_losses,
                pg_losses,
                entropy_losses,
                approx_kls,
                wc_trajes,
                act_trajes,
                adv_losses,
                reg_losses,
                clipfracs,
            ) = ([], [], [], [], [], [], [], [], [])
            eps = self.eps_schedule(current_t=self.current_step / self.config.total_timesteps)
            relax_beta = self.relax_beta_schedule(
                current_t=self.current_step / self.config.total_timesteps
            )
            for _ in range(self.config.n_epochs):
                if self.config.full_traj:
                    prev_log_rho = torch.zeros(b_obs.size(0))
                for start in range(0, b_obs.shape[1], self.config.batch_size):
                    end = start + self.config.batch_size
                    _, newlogprob, entropy, new_mean, *_ = self.actor(
                        obs=b_obs[:, start:end], action=b_actions[:, start:end]
                    )
                    logratio = newlogprob - b_logprobs[:, start:end]
                    ratio = logratio.exp()

                    mb_advantages = b_advantages[:, start:end]
                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.config.clip_range, 1 + self.config.clip_range
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2)[b_mask[:, start:end]].mean()
                    pg_losses.append(pg_loss.item())
                    entropy_loss = entropy[b_mask[:, start:end]].mean()
                    entropy_losses.append(entropy_loss.item())
                    loss = pg_loss - self.config.ent_coef * entropy_loss
                    #! START adv
                    if eps > 0:
                        lb_logprob, ub_logprob, reg_loss = self.relax_solver(
                            obs=b_obs[:, start:end],
                            eps=eps,
                            action=b_actions[:, start:end],
                            beta=relax_beta,
                            with_kl=(self.config.reg_coef > 0),
                            mu_0=new_mean,  # .detach(),
                        )
                        wc_logprob = torch.where(
                            org_b_advantage[:, start:end] >= 0, lb_logprob, ub_logprob
                        )
                        log_rho = wc_logprob - b_logprobs[:, start:end]
                        log_rho = log_rho.cumsum(dim=-1)
                        if self.config.clip_logrho:
                            log_rho = torch.clamp(
                                log_rho, min=self.config.min_logrho, max=self.config.max_logrho
                            )
                        log_rho = log_rho.masked_fill(~b_mask[:, start:end], float("-inf"))
                        if self.config.full_traj:
                            log_rho += prev_log_rho.unsqueeze(dim=-1)
                            prev_log_rho = log_rho[:, -1].clone().detach()
                        rho = log_rho.exp()
                        if self.config.discount_traj:
                            T = b_rewards[:, start:end].shape[-1]
                            expo = torch.arange(T)
                            if self.config.full_traj:
                                expo += start
                            gammas = self.config.gamma**expo
                            rewards = b_rewards[:, start:end] * gammas
                        else:
                            rewards = b_rewards[:, start:end]

                        wc_traj = (rho * rewards).sum() / b_mask[:, start].sum(dim=0)
                        act_traj = (rewards).sum() / b_mask[:, start].sum(dim=0)
                        adv_loss = self.adv_loss_fun(act_traj=act_traj, wc_traj=wc_traj)
                        wc_trajes.append(wc_traj.item())
                        act_trajes.append(act_traj.item())
                        adv_losses.append(adv_loss.item())
                        if self.config.conv_loss:
                            loss = (1 - self.config.rb_coef) * loss + self.config.rb_coef * adv_loss
                        else:
                            loss += self.config.rb_coef * adv_loss
                        if self.config.reg_coef > 0:
                            loss += self.config.reg_coef * reg_loss
                            reg_losses.append(reg_loss.item())

                    #! END adv
                    # Value loss
                    newvalue, _ = self.critic(obs=b_obs[:, start:end])
                    newvalue = newvalue.reshape(self.config.num_envs, -1)
                    if self.config.clip_range_vf > 0:
                        v_loss_unclipped = (newvalue - b_returns[:, start:end]) ** 2
                        v_clipped = b_values[:, start:end] + torch.clamp(
                            newvalue - b_values[:, start:end],
                            -self.config.clip_range_vf,
                            self.config.clip_range_vf,
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
                    if self.config.max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                        nn.utils.clip_grad_norm_(
                            self.critic.parameters(), self.config.max_grad_norm
                        )
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio)[b_mask[:, start:end]].mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.config.clip_range)
                            .float()[b_mask[:, start:end]]
                            .mean()
                            .item()
                        ]
                        approx_kls.append(approx_kl.item())
            y_pred, y_true = (
                b_values[b_mask].cpu().numpy(),
                b_returns[b_mask].cpu().numpy(),
            )
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
                "train/beta": relax_beta,
            }

            if self.config.anneal_lr:
                training_metrics.update(
                    {"train/lr_actor": lr_actor_now, "train/lr_critic": lr_critic_now}
                )
            if eps > 0 and self.config.rb_coef > 0:
                training_metrics.update(
                    {
                        "train/wc_traj": np.mean(wc_trajes),
                        "train/act_traj": np.mean(act_trajes),
                        "train/adv_loss": np.mean(adv_losses),
                    }
                )
                if self.config.reg_coef > 0:
                    training_metrics.update(
                        {
                            "train/reg_loss": np.mean(reg_losses),
                        }
                    )
            self.logger.log_metrics(metrics=training_metrics, step=self.current_step)
            if self.eval_flag:
                self.evaluate()
                self.eval_flag = False
        if self.config.checkpoint:
            self.save()
        self.evaluate()

    def get_which_adv_loss(self):
        if self.config.rb_loss == "relu":
            return lambda act_traj, wc_traj: F.relu(act_traj - wc_traj)
        elif self.config.rb_loss == "mse":
            return lambda act_traj, wc_traj: F.mse_loss(act_traj, wc_traj)
        elif self.config.rb_loss == "mae":
            return lambda act_traj, wc_traj: F.l1_loss(act_traj, wc_traj)
        elif self.config.rb_loss == "wc":
            return lambda act_traj, wc_traj: -wc_traj
        else:
            raise ValueError(f" Either rb_loss Not Implemented Yet: {self.config.rb_loss}")

    def batchify(self, episodes):
        for episode in episodes:
            for key, vals in episode.items():
                episode[key] = torch.stack(vals).float().to(self.device)
        lengths = [len(episode["obs"]) for episode in episodes]
        max_length = max(lengths)
        obs = torch.zeros(
            (self.config.num_envs, max_length) + self.env.single_observation_space.shape
        ).to(self.device)
        actions = torch.zeros(
            (self.config.num_envs, max_length) + self.env.single_action_space.shape
        ).to(self.device)
        logprobs = torch.zeros((self.config.num_envs, max_length)).to(self.device)
        rewards = torch.zeros((self.config.num_envs, max_length)).to(self.device)
        values = torch.zeros((self.config.num_envs, max_length)).to(self.device)
        mask = torch.zeros((self.config.num_envs, max_length)).to(self.device)
        for i in range(self.config.num_envs):
            obs[i, : lengths[i]] = episodes[i]["obs"]
            actions[i, : lengths[i]] = episodes[i]["actions"]
            logprobs[i, : lengths[i]] = episodes[i]["logprobs"]
            rewards[i, : lengths[i]] = episodes[i]["rewards"]
            values[i, : lengths[i]] = episodes[i]["values"]
            mask[i, : lengths[i]] = 1
        return (
            obs,
            actions,
            logprobs,
            rewards,
            values,
            mask.bool(),
        )

    def relax_solver(self, obs, action, eps, beta, with_kl=False, mu_0=None):
        num_envs, batch_size, act_dim = action.shape
        obs = obs.flatten(0, 1)
        action = action.flatten(0, 1)
        log_sigma = self.actor.logstd_head.expand_as(action).clone()  # .detach()
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
            if beta > 0:
                cr_x = None
            else:
                cr_x = (bounded_obs,)
            cr_lb, cr_ub = self.relaxed_actor.compute_bounds(x=cr_x, method="backward")
        else:
            cr_lb, cr_ub = 0, 0
        mu_lb = beta * ibp_lb + (1 - beta) * cr_lb
        mu_ub = beta * ibp_ub + (1 - beta) * cr_ub
        d_lb = (((action - torch.clamp(action, mu_lb, mu_ub)) / sigma) ** 2).sum(dim=-1)
        d_ub = ((torch.max(torch.abs(mu_lb - action), torch.abs(mu_ub - action)) / sigma) ** 2).sum(
            dim=-1
        )
        log_denom = 0.5 * action.shape[-1] * np.log(2 * np.pi) + torch.log(sigma).sum(dim=-1)
        lb_log_pi = -0.5 * d_ub - log_denom
        ub_log_pi = -0.5 * d_lb - log_denom
        if with_kl and mu_0 is not None:
            diff = torch.max(
                torch.abs(mu_lb.reshape(num_envs, batch_size, act_dim) - mu_0),
                torch.abs(mu_ub.reshape(num_envs, batch_size, act_dim) - mu_0),
            )
            kl = (((diff) / sigma.reshape_as(mu_0)) ** 2).sum(dim=-1).mean()
        else:
            kl = None
        return lb_log_pi.reshape(num_envs, batch_size), ub_log_pi.reshape(num_envs, batch_size), kl

    def evaluate(self):
        for i in range(self.config.num_eval_envs):
            self.eval_env.envs[i].set_wrapper_attr(
                "obs_rms", self.env.envs[i % self.config.num_envs].get_wrapper_attr("obs_rms")
            )
        eval_rewards = []
        eval_lengths = []
        obs, _ = self.eval_env.reset()
        alive_envs = self.eval_env.alive_envs
        while len(alive_envs) > 0:
            obs = torch.Tensor(obs).to(self.device)
            with torch.no_grad():
                action, _ = self.actor.get_action(obs=obs, deterministic=self.config.deterministic)
            obs, _, terminated, truncated, infos = self.eval_env.step(action.cpu().numpy())
            alive_envs = self.eval_env.alive_envs
            done = np.logical_or(terminated, truncated)
            obs = obs[~done]
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
        path = f"{self.workdir}/{self.config.env_id}/advis_pd/{self.run_name}"
        torch.save(self.actor.state_dict(), f"{path}/policy.pt")
        torch.save(self.critic.state_dict(), f"{path}/critic.pt")
        envparams, wrappers = env_metadata(self.env, self.config)
        with open(f"{path}/wrappers.json", "w") as f:
            json.dump(wrappers, f, indent=2)
        np.savez_compressed(f"{path}/envparams.npz", **envparams)

    def load(self):
        pass
