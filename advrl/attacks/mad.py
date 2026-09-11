import torch
import numpy as np


class MAD:
    """
    Maximal Action Difference
    Paper: (algo 3, p 27) Robust Deep Reinforcement Learning against Adversarial Perturbations on State Observations
    Arxiv: https://arxiv.org/abs/2003.08938
    """

    def __init__(self, config, agent, low, high, **kwargs):
        self.config = config
        self.agent = agent
        self.eps = self.config.eps
        self.low = low
        self.high = high

    def reset_hidden(self, index):
        return

    def poweriter_solver(self, obs, h, eps, mu_0, sigma):
        delta = torch.zeros_like(obs).normal_()
        for _ in range(self.config.pi.num_iter):
            delta = delta.clone().detach().requires_grad_(True)
            mu, *_ = self.agent.get_mu_and_std(obs=obs + self.config.pi.xi * delta, h=h)
            kl = (((mu - mu_0) / sigma) ** 2).sum(dim=-1)
            loss = kl
            loss = loss.sum()
            loss.backward()
            delta = eps * delta.grad.sign()
        return torch.clamp(obs + delta.detach().clone(), min=self.low, max=self.high)

    def sgld_solver(self, obs, h, eps, mu_0, sigma):
        obs_ = obs.clone()
        step_size = self.config.sgld.step_size
        if step_size < 0:
            step_size = eps / self.config.sgld.num_iter
        sgld_factor = np.sqrt(2 / (self.config.sgld.beta * step_size))
        for k in range(self.config.sgld.num_iter):
            obs = obs.clone().detach().requires_grad_(True)
            mu, *_ = self.agent.get_mu_and_std(obs=obs, h=h)
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
            obs = torch.clamp(obs, min=obs_ - eps, max=obs_ + eps)
            obs = torch.clamp(obs, min=self.low, max=self.high)
        return obs

    def get_perturbation(self, *, obs, h):
        with torch.no_grad():
            mu_0, sigma, _ = self.agent.get_mu_and_std(obs=obs, h=h)
        obs_0 = obs.clone()
        if self.config.solver == "sgld":
            obs = self.sgld_solver(obs, h, self.eps, mu_0, sigma)
        elif self.config.solver == "pi":
            obs = self.poweriter_solver(obs, h, self.eps, mu_0, sigma)
        else:
            raise ValueError(f"Solver {self.config.solver} not implemented yet")

        assert torch.abs(obs - obs_0).max().detach().item() <= self.eps + 1e-6
        return obs.clone().detach()
