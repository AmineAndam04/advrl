import torch


class CriticAttack:
    """
    Critic attack
    Paper: Robust Deep Reinforcement Learning against Adversarial Perturbations on State Observations
    Arxiv: https://arxiv.org/abs/2003.08938
    """

    def __init__(self, config, agent, low, high, **kwargs):
        self.config = config
        self.agent = agent
        self.num_iter = self.config.num_iter
        self.eps = self.config.eps
        self.step_size = self.config.step_size
        if self.step_size < 0:
            self.step_size = self.eps / self.num_iter
        self.low = low
        self.high = high
        self.hidden_critic = None

    def reset_hidden(self, index):
        if self.hidden_critic is not None:
            self.hidden_critic[0][:, index] = torch.zeros_like(self.hidden_critic[0][:, index])
            self.hidden_critic[1][:, index] = torch.zeros_like(self.hidden_critic[1][:, index])

    def get_perturbation(self, *, obs, h=None):
        obs_0 = obs.clone()
        for k in range(self.num_iter):
            obs = obs.clone().detach().requires_grad_(True)
            if k == 0:
                # To update hidden_critic
                value, hidden_critic = self.agent(obs=obs, h=self.hidden_critic)

            else:
                value, *_ = self.agent(obs=obs, h=self.hidden_critic)
            loss = value.sum()
            loss.backward()
            obs = obs - self.step_size * obs.grad.sign()
            obs = torch.clamp(obs, min=obs_0 - self.eps, max=obs_0 + self.eps)
            obs = torch.clamp(obs, min=self.low, max=self.high)
        if hidden_critic is not None:
            self.hidden_critic = (
                hidden_critic[0].clone().detach(),
                hidden_critic[1].clone().detach(),
            )
        assert torch.abs(obs - obs_0).max().detach().item() <= self.eps + 1e-6
        return obs.clone().detach()
