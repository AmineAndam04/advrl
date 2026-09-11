from ml_collections import ConfigDict
import math


def get_config():
    config = ConfigDict()
    config.attack_name = "paad"
    # Attack params
    config.num_iter = 10  # PGD iterations
    config.step_size = -1.0  # PGD's step size , if step_size<0 we use step_size=eps/num_iter
    config.eps = 0.15
    config.deterministic = True
    # Evaluation params
    config.num_envs = 10
    config.num_episodes = 50
    config.num_seeds = 20
    config.auto_seeds = True
    config.reprd_auto_seeds = 0
    config.pre_seeds = ""
    config.device = "cpu"
    # Training params
    config.train = True  # If True, train the adversary
    config.pretrain_path = ""  # Path to load the adversary
    training = ConfigDict()
    training.env_id = (
        None  # Leave it as None. Will be automatically fetched from the agent configs.
    )
    training.total_timesteps = 1000000
    training.actor = "pi_mlp"  # Registry
    training.critic = "vf_mlp"  # Registry
    training.actor_lstm = False
    training.critic_lstm = False  # it's considered only when actor is lstm
    training.pi_net = (64, 64)
    training.vf_net = (64, 64)
    training.pi_act_fun = "tanh"
    training.vf_act_fun = "tanh"
    training.log_std_init = -1.0
    training.ortho_init = True
    training.orth_std = math.sqrt(2)
    training.lr_actor = 0.003
    training.lr_critic = 0.0005
    training.num_envs = 2  # Number of parallel environments
    training.n_steps = 2048
    training.batch_size = 64
    training.n_epochs = 10
    training.optim = "Adam"
    training.anneal_lr = False
    training.min_lr = 0.0000001
    training.gamma = 0.99
    training.gae_lambda = 0.95
    training.clip_range = 0.2
    training.clip_range_vf = 0.2
    training.normalize_advantage = True
    training.normalize_return = True
    training.ent_coef = 0.00
    training.max_grad_norm = -1.0
    training.num_iter = 2  # PAAD number of iterations
    training.step_size = -1.0  # PAAD step size, if <0, we use step_size=epsilon/num_iter
    training.eps_schedule = "constant"  # Scheduler epsilon: constant, linear, cycle
    training.eps = 0.15  # with constant scheduler
    training.start_val = 0.0  # linear or cycle
    training.end_val = 0.15  # linear or cycle
    training.start_t = 0.0  # linear or cycle
    training.end_t = 0.75  # linear or cycle
    training.max_val = 0.15  # cycle
    training.max_t = 0.75  # cycle
    training.seed = 0
    training.checkpoint = False
    training.log_every = 10
    if config.train:
        config.training = training
    # Logging
    config.logger = "tensorboard"  # Options: "tensorboard", "wandb", "aim", or "console"
    config.project_name = "advrl"
    config.repo = ".aim"  # for aim
    config.experiment = "paad"  # for aim
    config.aim_tag = "v1"
    return config
