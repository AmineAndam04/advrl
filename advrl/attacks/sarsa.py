import os
import json
import torch
from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gymnasium.spaces import flatdim
from advrl.utils.misc import set_random_seed, set_scheduler
from advrl.envs.mujoco import create_make_env_mujoco_from_json
from advrl.envs.vec_env import VecEnv
from advrl.utils.nets import orth_init, no_init
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from ml_collections import ConfigDict

# TODO: LSTM based SARSA

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}


class SarsaCritic(nn.Module):
    def __init__(
        self,
        in_dim: int,
        net_arch: tuple,
        ortho_init: bool,
        act_fun: str,
        orth_std: float,
    ) -> None:
        super().__init__()
        layer_init = orth_init if ortho_init else no_init
        act_fun = ACTIVATIONS[act_fun]
        critic = [
            layer_init(nn.Linear(in_dim, net_arch[0]), std=orth_std),
            act_fun(),
        ]
        for i in range(len(net_arch) - 1):
            critic.append(layer_init(nn.Linear(net_arch[i], net_arch[i + 1]), std=orth_std))
            critic.append(act_fun())
        critic.append(layer_init(nn.Linear(net_arch[-1], 1), std=1.0))
        self.critic = nn.Sequential(*critic)

    def forward(self, x):
        return self.critic(x)

    # def reset_h(self):
    #     self.h = None

    # def detach_h(self):
    #     if self.h is not None:
    #         self.h = (self.h[0].detach(), self.h[1].detach())


class SarsaAttack:
    """
    Robust SARSA attack
    Paper: (algo 4, p 27) Robust Deep Reinforcement Learning against Adversarial Perturbations on State Observations
    Arxiv: https://arxiv.org/abs/2003.08938
    """

    def __init__(self, config, agent, low, high, **kwargs):
        self.config = config
        self.agent = agent
        self.num_iter = self.config.num_iter
        self.eps = self.config.eps
        if self.config.step_size < 0:
            self.step_size = self.eps / self.num_iter
        else:
            self.step_size = self.config.step_size
        self.low = low
        self.high = high
        self.device = self.config.device
        if not config.train:
            self.load()

    def reset_hidden(self, index):
        return

    def _setup(self):
        # Initialize the environment
        set_random_seed(self.cfg_train.seed)
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
        self.critic = SarsaCritic(
            in_dim=flatdim(self.env.single_observation_space)
            + flatdim(self.env.single_action_space),
            net_arch=self.cfg_train.net_arch,
            ortho_init=self.cfg_train.ortho_init,
            act_fun=self.cfg_train.act_fun,
            orth_std=self.cfg_train.orth_std,
        )
        self.target_critic = deepcopy(self.critic)
        optimizer = getattr(optim, self.cfg_train.optim)
        self.optimizer = optimizer(self.critic.parameters(), lr=self.cfg_train.lr)

        os.makedirs(self.save_path, exist_ok=True)
        self.run_name = "Sarsa"
        self.logger.run_name = self.run_name
        self.logger.initialize_logger()
        # Scheduler
        if self.cfg_train.rb_coef > 0:
            self.eps_schedule = set_scheduler(config=self.cfg_train)
            self.relax_beta_schedule = set_scheduler(
                config=self.cfg_train.relax, var_to_scheduler="beta"
            )
            self.relaxed_critic = BoundedModule(
                self.critic,
                torch.randn(
                    1,
                    flatdim(self.env.single_observation_space)
                    + flatdim(self.env.single_action_space),
                ),
            )

    def collect_nsteps(
        self,
    ):
        if self.current_step == 0:
            observation, _ = self.env.reset()
            observation = torch.Tensor(observation).to(self.device)
            h = None
        else:
            observation = self.last_obs
            h = self.last_h

        obs = torch.zeros(
            (self.cfg_train.n_steps + 1, self.cfg_train.num_envs)
            + self.env.single_observation_space.shape
        ).to(self.device)
        actions = torch.zeros(
            (self.cfg_train.n_steps + 1, self.cfg_train.num_envs)
            + self.env.single_action_space.shape
        ).to(self.device)
        rewards = torch.zeros((self.cfg_train.n_steps + 1, self.cfg_train.num_envs)).to(self.device)
        dones = torch.zeros((self.cfg_train.n_steps + 1, self.cfg_train.num_envs)).to(self.device)
        for step in range(0, self.cfg_train.n_steps + 1):
            if self.cfg_train.adv_steps:
                observation = self.get_perturbation(obs=observation, h=h)
            with torch.no_grad():
                action, h = self.agent.get_action(
                    obs=observation, deterministic=self.config.deterministic, h=h
                )
                action = action.reshape(
                    (self.cfg_train.num_envs,) + self.env.single_action_space.shape
                )
            next_obs, reward, terminations, truncations, infos = self.env.step(action.cpu().numpy())
            done = np.logical_or(terminations, truncations)
            done = torch.Tensor(done).to(self.device)
            obs[step] = observation
            rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            dones[step] = done
            actions[step] = action
            observation = torch.Tensor(next_obs).to(self.device)
            self.current_step += self.cfg_train.num_envs
            for i, info in enumerate(infos):
                if "episode" in info:
                    if h is not None:
                        h[0][:, i] = 0
                        h[1][:, i] = 0
                    ep_rd = info["episode"]["r"]
                    ep_len = info["episode"]["l"]
                    self.traj_rewards.append(ep_rd)
                    self.traj_lengths.append(ep_len)
                    if len(self.traj_rewards) == self.cfg_train.log_every:
                        self.logger.log_metrics(
                            metrics={
                                "traj/rewards": np.mean(self.traj_rewards),
                                "traj/ep_lengths": np.mean(self.traj_lengths),
                            },
                            step=self.current_step,
                        )
                        self.traj_rewards = []
                        self.traj_lengths = []

        self.last_obs = observation
        self.last_h = h
        b_obs = obs[:-1].reshape((-1,) + self.env.single_observation_space.shape)
        b_next_obs = obs[1:].reshape((-1,) + self.env.single_observation_space.shape)
        b_actions = actions[:-1].reshape((-1,) + self.env.single_action_space.shape)
        b_next_actions = actions[1:].reshape((-1,) + self.env.single_action_space.shape)
        b_rewards = rewards[:-1].reshape(-1)
        b_dones = dones[:-1].reshape(-1)
        return b_obs, b_next_obs, b_actions, b_next_actions, b_rewards, b_dones

    def train(self, cfg_train, wrappers, envparams, workdir, save_path, logger):
        self.cfg_train = cfg_train
        self.wrappers = wrappers
        self.envparams = envparams
        self.workdir = workdir
        self.save_path = save_path
        self.logger = logger
        self.traj_rewards = []
        self.traj_lengths = []
        self._setup()
        self.current_step = 0
        self.num_updates = 0
        while self.current_step < self.cfg_train.total_timesteps:
            b_obs, b_next_obs, b_action, b_next_action, b_rewards, b_dones = self.collect_nsteps()
            td_losses = []
            rb_losses = []
            losses = []
            if self.cfg_train.rb_coef > 0:
                eps = self.eps_schedule(
                    current_t=self.current_step / self.cfg_train.total_timesteps
                )
                relax_beta = self.relax_beta_schedule(
                    current_t=self.current_step / self.cfg_train.total_timesteps
                )
            for start in range(0, self.cfg_train.n_steps, self.cfg_train.batch_size):
                end = start + self.cfg_train.batch_size
                with torch.no_grad():
                    next_obs_action = torch.cat(
                        [b_next_obs[start:end], b_next_action[start:end]], dim=-1
                    )
                    next_q = self.target_critic(next_obs_action).squeeze()
                    td_target = (
                        b_rewards[start:end]
                        + (1 - b_dones[start:end]) * self.cfg_train.gamma * next_q
                    )
                obs_action = torch.cat([b_obs[start:end], b_action[start:end]], dim=-1)
                q_values = self.critic(obs_action).squeeze()
                td_loss = ((q_values - td_target) ** 2).mean()
                td_losses.append(td_loss.item())
                if self.cfg_train.rb_coef > 0:
                    rb_loss = self.relax_solver(
                        obs_action=obs_action, eps=eps, beta=relax_beta, q_values=q_values
                    )

                    rb_losses.append(rb_loss.item())
                    loss = td_loss + self.cfg_train.rb_coef * rb_loss
                else:
                    loss = td_loss
                losses.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                if self.cfg_train.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg_train.max_grad_norm)
                self.optimizer.step()
            self.num_updates += 1
            if self.num_updates % self.cfg_train.update_freq == 0:
                self.soft_update(
                    target_net=self.target_critic,
                    critic_net=self.critic,
                    polyak=self.cfg_train.polyak,
                )
            metrics = {"train/loss": np.mean(losses)}
            if self.cfg_train.rb_coef > 0:
                metrics.update(
                    {
                        "train/td_loss": np.mean(td_losses),
                        "train/rb_loss": np.mean(rb_losses),
                        "train/eps": eps,
                        "train/beta": relax_beta,
                    }
                )
            self.logger.log_metrics(metrics=metrics, step=self.current_step)
        self.save()

    def get_perturbation(self, *, obs, h):
        obs_0 = obs.clone()
        for _ in range(self.num_iter):
            obs = obs.clone().detach().requires_grad_(True)
            action, *_ = self.agent.get_action(
                obs=obs, h=h, deterministic=self.config.deterministic
            )
            action = action.reshape((obs.shape[0], -1))
            obs_action = torch.cat([obs_0, action], dim=-1)
            q_value = self.critic(obs_action)
            loss = q_value.sum()
            loss.backward()
            obs = obs - self.step_size * obs.grad.sign()
            obs = torch.clamp(obs, min=obs_0 - self.eps, max=obs_0 + self.eps)
            obs = torch.clamp(obs, min=self.low, max=self.high)
        assert torch.abs(obs - obs_0).max().detach().item() <= self.eps + 1e-6
        return obs.clone().detach()

    def load(self):
        with open(f"{self.config.pretrain_path}/config.json") as f:
            config = ConfigDict(json.load(f))
        state_dict = torch.load(f"{self.config.pretrain_path}/model.pt", map_location="cpu")
        self.critic = SarsaCritic(
            in_dim=config.in_dim,
            net_arch=config.net_arch,
            ortho_init=config.ortho_init,
            orth_std=config.orth_std,
        ).to(self.device)
        self.critic.load_state_dict(state_dict)

    def relax_solver(self, obs_action, eps, beta, q_values):
        ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
        bounded_obs_action = BoundedTensor(obs_action, ptb)
        # IBP bounds
        if beta > 0:
            ibp_lb, ibp_ub = self.relaxed_critic.compute_bounds(
                x=(bounded_obs_action,), method="ibp"
            )
        else:
            ibp_lb, ibp_ub = 0, 0
        # CROWN-IBP
        if (1 - beta) > 0:
            if beta > 0:
                cr_x = None
            else:
                cr_x = (bounded_obs_action,)
            cr_lb, cr_ub = self.relaxed_critic.compute_bounds(x=cr_x, method="backward")
        else:
            cr_lb, cr_ub = 0, 0

        lb = beta * ibp_lb + (1 - beta) * cr_lb
        ub = beta * ibp_ub + (1 - beta) * cr_ub
        lb, ub = lb.squeeze(), ub.squeeze()
        rb_loss = (torch.max(lb - q_values, ub - q_values) ** 2).mean()
        return rb_loss

    def soft_update(self, target_net, critic_net, polyak):
        for target_param, param in zip(target_net.parameters(), critic_net.parameters()):
            target_param.data.copy_(polyak * param.data + (1.0 - polyak) * target_param.data)

    def save(self):
        path = f"{self.save_path}/{self.run_name}"
        os.makedirs(path, exist_ok=True)
        if self.cfg_train.checkpoint:
            torch.save(self.critic.state_dict(), f"{path}/model.pt")
        config = self.cfg_train.to_dict()
        config.update(
            {
                "in_dim": flatdim(self.env.single_observation_space)
                + flatdim(self.env.single_action_space)
            }
        )
        with open(f"{path}/config.json", "w") as f:
            json.dump(config, f, indent=2)
