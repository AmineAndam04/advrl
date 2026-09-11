import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from advrl.utils.nets import orth_init, no_init
from ml_collections import ConfigDict

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


class ActorMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, net_arch: ConfigDict, squash: bool = False) -> None:
        """
        I use h in the arguments and return so that I can write one evaluation and attack script for both LSTM and MLP based
        """
        super().__init__()
        self.squash = squash
        # initialization
        layer_init = orth_init if net_arch.ortho_init else no_init
        self.act_fun = ACTIVATIONS[net_arch.act_fun]
        # mean head
        mean_head = [
            layer_init(nn.Linear(in_dim, net_arch.pi_net[0]), std=net_arch.orth_std),
            self.act_fun(),
        ]
        for i in range(len(net_arch.pi_net) - 1):
            mean_head.append(
                layer_init(nn.Linear(net_arch.pi_net[i], net_arch.pi_net[i + 1]), std=net_arch.orth_std)
            )
            mean_head.append(self.act_fun())
        mean_head.append(layer_init(nn.Linear(net_arch.pi_net[-1], out_dim), std=0.01))
        self.mean_head = nn.Sequential(*mean_head)
        # log std head
        self.logstd_head = nn.Parameter(torch.zeros(1, out_dim) + net_arch.log_std_init)

    def forward(self, obs, action=None, h=None):
        action_mean = self.mean_head(obs)
        action_logstd = self.logstd_head.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            if self.squash:
                unsquashed_action = probs.sample()
                action = torch.tanh(unsquashed_action)
                log_unsquashed = probs.log_prob(unsquashed_action).sum(-1)
                log_prob = log_unsquashed - torch.log(1 - action**2 + 1e-6).sum(-1)
            else:
                action = probs.sample()
                log_prob = probs.log_prob(action).sum(-1)
        else:
            if self.squash:
                unsquashed_action = torch.atanh(torch.clamp(action, -1 + 1e-6, 1 - 1e-6))
                log_unsquashed = probs.log_prob(unsquashed_action).sum(-1)
                log_prob = log_unsquashed - torch.log(1 - action**2 + 1e-6).sum(-1)
            else:
                log_prob = probs.log_prob(action).sum(-1)
        return action, log_prob, probs.entropy().sum(-1), action_mean, None

    def get_action(self, obs, deterministic=True, h=None):
        action_mean = self.mean_head(obs)
        if deterministic:
            if self.squash:
                return torch.tanh(action_mean), None
            else:
                return action_mean, None
        action_logstd = self.logstd_head.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        action = probs.rsample()
        if self.squash:
            action = torch.tanh(action)
        return action, None

    def get_mu_and_std(self, obs, h=None):
        action_mean = self.mean_head(obs)
        action_logstd = self.logstd_head.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return action_mean, action_std, h


class RelaxedActorMLP(nn.Module):
    def __init__(self, mean_head):
        super().__init__()
        self.mean_head = mean_head

    def forward(self, x):
        action_mean = self.mean_head(x)
        return action_mean


class ActorLSTM(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, net_arch: ConfigDict, squash: bool = False) -> None:
        super().__init__()
        self.squash = squash
        self.lstm_dim = net_arch.pi_lstm_dim
        # initialization
        layer_init = orth_init if net_arch.ortho_init else no_init
        # mean head
        self.embed = layer_init(nn.Linear(in_dim, self.lstm_dim), std=net_arch.orth_std)
        self.lstm = nn.LSTM(self.lstm_dim, self.lstm_dim, batch_first=True)
        self.mean = layer_init(nn.Linear(self.lstm_dim, out_dim), std=1)

        # log std head
        self.log_std = nn.Parameter(torch.zeros(1, out_dim) + net_arch.log_std_init)

    def forward(self, obs, action=None, h=None):
        if h is None:
            h = (
                torch.zeros(1, obs.size(0), self.lstm_dim).to(obs.device),
                torch.zeros(1, obs.size(0), self.lstm_dim).to(obs.device),
            )
        if obs.dim() < 3:
            obs = obs.unsqueeze(1)
        x = self.embed(obs)
        x, h = self.lstm(x, h)
        mean = self.mean(x)
        std = torch.exp(self.log_std.expand_as(mean))
        probs = Normal(mean, std)
        if action is None:
            if self.squash:
                unsquashed_action = probs.sample()
                action = torch.tanh(unsquashed_action)
                log_unsquashed = probs.log_prob(unsquashed_action).sum(-1)
                log_prob = log_unsquashed - torch.log(1 - action**2 + 1e-6).sum(-1)
            else:
                action = probs.sample()
                log_prob = probs.log_prob(action).sum(-1)
        else:
            if self.squash:
                unsquashed_action = torch.atanh(torch.clamp(action, -1 + 1e-6, 1 - 1e-6))
                log_unsquashed = probs.log_prob(unsquashed_action).sum(-1)
                log_prob = log_unsquashed - torch.log(1 - action**2 + 1e-6).sum(-1)
            else:
                log_prob = probs.log_prob(action).sum(-1)
        return action, log_prob, probs.entropy().sum(-1), mean, h

    def get_mu_and_std(self, obs, h=None):
        if h is None:
            h = (
                torch.zeros(1, obs.size(0), self.lstm_dim).to(obs.device),
                torch.zeros(1, obs.size(0), self.lstm_dim).to(obs.device),
            )
        if obs.dim() < 3:
            obs = obs.unsqueeze(1)
        x = self.embed(obs)
        x, h = self.lstm(x, h)
        mean = self.mean(x)
        std = torch.exp(self.log_std.expand_as(mean))
        return mean, std, h

    def get_action(self, obs, h=None, deterministic=True):
        if h is None:
            h = (
                torch.zeros(1, obs.size(0), self.lstm_dim).to(obs.device),
                torch.zeros(1, obs.size(0), self.lstm_dim).to(obs.device),
            )
        if obs.dim() < 3:
            obs = obs.unsqueeze(1)
        x = self.embed(obs)
        x, h = self.lstm(x, h)
        mean = self.mean(x)
        if deterministic:
            if self.squash:
                return torch.tanh(mean), h
            else:
                return mean, h
        std = torch.exp(self.log_std.expand_as(mean))
        probs = Normal(mean, std)
        action = probs.rsample()
        if self.squash:
            action = torch.tanh(action)
        return action, h


class CriticMLP(nn.Module):
    def __init__(self, in_dim: int, net_arch: ConfigDict):
        super().__init__()
        layer_init = orth_init if net_arch.ortho_init else no_init
        self.act_fun = ACTIVATIONS[net_arch.act_fun]
        critic = [
            layer_init(nn.Linear(in_dim, net_arch.vf_net[0]), std=net_arch.orth_std),
            self.act_fun(),
        ]
        for i in range(len(net_arch.vf_net) - 1):
            critic.append(
                layer_init(nn.Linear(net_arch.vf_net[i], net_arch.vf_net[i + 1]), std=net_arch.orth_std)
            )
            critic.append(self.act_fun())
        critic.append(layer_init(nn.Linear(net_arch.vf_net[-1], 1), std=1.0))
        self.critic = nn.Sequential(*critic)

    def forward(self, obs, h=None):
        return self.critic(obs), h


class CriticLSTM(nn.Module):
    def __init__(self, in_dim: int, net_arch: ConfigDict):
        super().__init__()
        self.lstm_dim = net_arch.vf_lstm_dim
        layer_init = orth_init if net_arch.ortho_init else no_init

        self.embed = layer_init(nn.Linear(in_dim, self.lstm_dim), std=net_arch.orth_std)
        self.lstm = nn.LSTM(self.lstm_dim, self.lstm_dim, batch_first=True)
        self.value = layer_init(nn.Linear(self.lstm_dim, 1), std=1)

    def forward(self, obs, h=None):
        if h is None:
            h = (
                torch.zeros(1, obs.size(0), self.lstm_dim).to(obs.device),
                torch.zeros(1, obs.size(0), self.lstm_dim).to(obs.device),
            )
        if obs.dim() < 3:
            obs = obs.unsqueeze(1)
        x = self.embed(obs)
        x, h = self.lstm(x, h)
        value = self.value(x)
        return value, h


class QnetworkContinuousMLP(nn.Module):
    def __init__(self, in_dim: int, net_arch: ConfigDict):
        super().__init__()
        layer_init = orth_init if net_arch.ortho_init else no_init
        self.act_fun = ACTIVATIONS[net_arch.act_fun]
        q_net = [
            layer_init(nn.Linear(in_dim, net_arch.q_net[0]), std=net_arch.orth_std),
            self.act_fun(),
        ]
        for i in range(len(net_arch.q_net) - 1):
            q_net.append(
                layer_init(nn.Linear(net_arch.q_net[i], net_arch.q_net[i + 1]), std=net_arch.orth_std)
            )
            q_net.append(self.act_fun())
        q_net.append(layer_init(nn.Linear(net_arch.q_net[-1], 1), std=0.1))
        self.q_net = nn.Sequential(*q_net)

    def forward(self, obs_action, h=None):
        return self.q_net(obs_action), h
