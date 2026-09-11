from ml_collections import ConfigDict
import math


def get_config():
    config = ConfigDict()
    config.name = "ppo"  # config id
    # Neural nets
    config.actor = "pi_mlp"  # Registry
    config.critic = "vf_mlp"  # Registry
    config.pi_net = (64, 64)
    config.vf_net = (64, 64)
    config.pi_act_fun = "tanh"
    config.vf_act_fun = "tanh"
    config.log_std_init = -2.0
    config.ortho_init = True
    config.orth_std = math.sqrt(2)
    # Env
    config.env_id = "HalfCheetah-v5"
    config.norm_obs = True
    config.clip_obs = True
    config.range_obs = (-10, 10)
    config.norm_reward = True
    config.clip_reward = True
    config.range_reward = (-10, 10)
    # Training
    config.num_envs = 2  # Number of parallel environments
    config.total_timesteps = 2000000
    config.n_steps = 2048
    config.batch_size = 64
    config.n_epochs = 10
    config.optim = "Adam"
    config.lr_actor = 0.00005
    config.lr_critic = 0.00005
    config.anneal_lr = False
    config.min_lr = 0.0000001
    config.gamma = 0.99
    config.gae_lambda = 0.95
    config.clip_range = 0.2
    config.clip_range_vf = 0.2
    config.normalize_advantage = True
    config.normalize_return = False
    config.ent_coef = 0.0
    config.max_grad_norm = -1.0
    config.seed = 0
    config.device = "cpu"
    # Logging
    config.log_every = 10
    config.exp_name = ""
    config.logger = "tensorboard"  # Options: "tensorboard", "wandb", "aim", or "console"
    config.project_name = "advrl"
    config.repo = ".aim"  # for aim
    config.experiment = "train"  # for aim
    config.verbose = 0
    config.algo_name = "ppo"
    config.prefix_run_name = "PPO"
    # Evaluation
    config.num_eval_envs = 1
    config.num_eval_episodes = 10
    config.eval_every = 200  # evaluate every 'eval_every' episode
    config.deterministic = True
    # Save
    config.checkpoint = True
    config.lock()
    return config
