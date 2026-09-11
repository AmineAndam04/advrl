from ml_collections import ConfigDict
import math


def get_config():
    config = ConfigDict()
    config.name = "atla_ppo"  # config id
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
    #! Adversarial training
    config.atla = ConfigDict()
    config.atla.actor = "pi_mlp"  # Registry
    config.atla.critic = "vf_mlp"  # Registry
    config.atla.squash = False
    config.atla.pi_net = (64, 64)
    config.atla.vf_net = (64, 64)
    config.atla.pi_act_fun = "tanh"
    config.atla.vf_act_fun = "tanh"
    config.atla.log_std_init = -2.0
    config.atla.ortho_init = True
    config.atla.orth_std = math.sqrt(2)
    config.atla.lr_actor = 0.00005
    config.atla.lr_critic = 0.00005
    config.atla.num_envs = 2  # Number of parallel environments
    config.atla.n_steps = 2048
    config.atla.batch_size = 64
    config.atla.n_epochs = 10
    config.atla.optim = "Adam"
    config.atla.anneal_lr = False
    config.atla.min_lr = 0.0000001
    config.atla.gamma = 0.99
    config.atla.gae_lambda = 0.95
    config.atla.clip_range = 0.2
    config.atla.clip_range_vf = 0.2
    config.atla.normalize_advantage = True
    config.atla.normalize_return = False
    config.atla.ent_coef = 0.0
    config.atla.max_grad_norm = -1.0
    config.atla.adv_scheduled = True  # If True the epsilon scheduler will be applied to both the agent and the adversary. If False only the agent
    config.atla.eps_schedule = "linear"  # Scheduler epsilon: constant, linear, cycle
    config.atla.eps = 0.15  # with constant scheduler
    config.atla.start_val = 0.0  # linear or cycle
    config.atla.end_val = 0.15  # linear or cycle
    config.atla.start_t = 0.0  # linear or cycle
    config.atla.end_t = 0.75  # linear or cycle
    config.atla.max_val = 0.15  # cycle
    config.atla.max_t = 0.75  # cycle
    # SA-reg
    config.solver = "sgld"  # sgld or pi
    config.sgld = ConfigDict({"beta": 1.0, "step_size": -1.0, "num_iter": 10})
    config.pi = ConfigDict({"num_iter": 1, "xi": 0.000001})
    config.reg_coef = -1.0

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
    config.algo_name = "atla"
    config.prefix_run_name = "ATLA-PPO"
    # Evaluation
    config.num_eval_envs = 10
    config.num_eval_episodes = 10
    config.eval_every = 200  # evaluate every 'eval_every' episode
    config.deterministic = True
    # Save
    config.checkpoint = True
    config.lock()
    return config
