from ml_collections import ConfigDict


def get_config():

    config = ConfigDict()
    config.attack_name = "atla"
    # Attack params
    config.eps = 0.15  # Attack budget
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
    training.pi_act_fun = "tanh"
    training.vf_act_fun = "tanh"
    training.actor_lstm = False
    training.critic_lstm = False  # it's considered only when actor is lstm
    training.pi_net = (64, 64)
    training.vf_net = (64, 64)
    training.squash = True  # Actions in [-1,1], true action is epsilon*[-1,1]
    training.log_std_init = -2.0
    training.ortho_init = True
    training.orth_std = 0.5**2
    training.lr_actor = 0.00005
    training.lr_critic = 0.00005
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
    training.ent_coef = 0.0
    training.max_grad_norm = -1.0
    training.eps_schedule = "constant"  # Scheduler epsilon: constant, linear, cycle
    training.eps = 0.15  # with constant scheduler
    training.start_val = 0.0  # linear or cycle
    training.end_val = 0.15  # linear or cycle
    training.start_t = 0.0  # linear or cycle
    training.end_t = 0.75  # linear or cycle
    training.max_val = 0.75  # cycle
    training.max_t = 0.75  # cycle
    training.seed = 0
    training.log_every = 10
    training.checkpoint = False
    if config.train:
        config.training = training

    config.logger = "tensorboard"  # Options: "tensorboard", "wandb", "aim", or "console"
    config.project_name = "advrl"
    config.repo = ".aim"  # for aim
    config.experiment = "atla"  # for aim
    config.aim_tag = "v1"
    return config
