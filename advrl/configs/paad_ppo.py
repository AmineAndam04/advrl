from ml_collections import ConfigDict
import math


def get_config():
    config = ConfigDict()
    config.name = "paad_ppo"  # config id
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
    config.paad = ConfigDict()
    config.paad.actor = "pi_mlp"  # Registry
    config.paad.critic = "vf_mlp"  # Registry
    config.paad.pi_net = (64, 64)
    config.paad.vf_net = (64, 64)
    config.paad.pi_act_fun = "tanh"
    config.paad.vf_act_fun = "tanh"
    config.paad.log_std_init = -2.0
    config.paad.ortho_init = True
    config.paad.orth_std = math.sqrt(2)
    config.paad.lr_actor = 0.00005
    config.paad.lr_critic = 0.00005
    config.paad.num_envs = 2  # Number of parallel environments
    config.paad.n_steps = 2048
    config.paad.batch_size = 64
    config.paad.n_epochs = 10
    config.paad.optim = "Adam"
    config.paad.anneal_lr = False
    config.paad.min_lr = 0.0000001
    config.paad.gamma = 0.99
    config.paad.gae_lambda = 0.95
    config.paad.clip_range = 0.2
    config.paad.clip_range_vf = 0.2
    config.paad.normalize_advantage = True
    config.paad.normalize_return = False
    config.paad.ent_coef = 0.0
    config.paad.max_grad_norm = -1.0
    config.paad.num_iter = 1  # PGD number of iterations
    config.paad.step_size = -1.0  # PGD step size, if <0, we use step_size=epsilon/num_iter
    config.paad.adv_scheduled = True  # If True the epsilon scheduler will be applied to both the agent and the adversary. If False only the agent
    config.paad.eps_schedule = "linear"  # Scheduler epsilon: constant, linear, cycle
    config.paad.eps = 0.15  # with constant scheduler
    config.paad.start_val = 0.0  # linear or cycle
    config.paad.end_val = 0.15  # linear or cycle
    config.paad.start_t = 0.0  # linear or cycle
    config.paad.end_t = 0.75  # linear or cycle
    config.paad.max_val = 0.75  # cycle
    config.paad.max_t = 0.75  # cycle
    config.paad.deterministic_victim = True
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
    config.algo_name = "paad"
    config.prefix_run_name = "PAAD-PPO"
    # Evaluation
    config.num_eval_envs = 10
    config.num_eval_episodes = 10
    config.eval_every = 200  # evaluate every 'eval_every' episode
    config.deterministic = True
    # Save
    config.checkpoint = True
    config.lock()
    return config
