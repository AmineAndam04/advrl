import torch


class RandomAttack:
    """
    Random attacks
    """

    def __init__(self, config, low, high, **kwargs):
        self.config = config
        self.attack_state = None
        self.eps = self.config.eps
        self.low = low
        self.high = high

    def reset_hidden(self, index):
        return

    def get_perturbation(self, *, obs, **kwargs):
        "obs is expected to be of shape (num_envs,obs_dim)"
        assert obs.ndim == 2
        num_envs, obs_dim = obs.shape
        if self.config.dist == "uniform":
            if self.config.signed:
                delta = torch.empty_like(obs).uniform_(-1, 1)
                delta = self.eps * delta.sign()
            else:
                delta = torch.empty_like(obs).uniform_(-self.eps, self.eps)
        elif self.config.dist == "gaussian":
            if self.config.type == "normal":
                delta = torch.distributions.Normal(0.0, 1.0).sample((num_envs, obs_dim))
                delta = self.eps * delta.sign()
            if self.config.type == "multivariate":
                mean = torch.zeros(obs_dim)
                A = torch.randn(obs_dim, obs_dim)
                AA_t = A @ A.t()
                cov = AA_t / (AA_t.max()) + 1e-6 * torch.eye(obs_dim)
                dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
                delta = dist.sample((num_envs,))
                delta = self.eps * delta.sign()
        elif self.config.dist == "fixed":
            if self.attack_state is None:
                if self.config.type == "normal":
                    delta = torch.distributions.Normal(0.0, 1.0).sample((num_envs, obs_dim))
                    delta = self.eps * delta.sign()
                elif self.config.type == "uniform":
                    delta = torch.empty_like(obs).uniform_(-1, 1)
                    delta = self.eps * delta.sign()
                elif self.config.type == "all_positive":
                    delta = self.eps * torch.ones_like(obs)
                elif self.config.type == "all_negatives":
                    delta = -self.eps * torch.ones_like(obs)
                self.attack_state = delta
            delta = self.attack_state
        elif self.config.dist == "discrete":
            x = torch.tensor([-1, 1], dtype=obs.dtype)
            delta = self.eps * x[torch.randint(low=0, high=2, size=obs.shape)]
        delta = torch.clamp(delta, min=-self.eps, max=self.eps)
        adv_obs = obs + delta
        adv_obs = torch.clamp(adv_obs, min=self.low, max=self.high)
        return adv_obs
