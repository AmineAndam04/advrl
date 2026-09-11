from ml_collections import ConfigDict


def get_config():

    config = ConfigDict()
    config.attack_name = "sarsa"
    # Attack params
    config.num_iter = 10  # PGD number of iterations with learned Q(s,a)
    config.step_size = -1.0  # PGD step size, if <0, we use step_size=epsilon/num_iter
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
    config.train = True  # If True, train the Q(s,a) from scratch
    config.pretrain_path = ""  # Path to load the pre-trained Q(s,a)
    training = ConfigDict()
    training.env_id = (
        None  # Leave it as None. Will be automatically fetched from the agent configs.
    )
    training.num_envs = 1
    training.net_arch = (64, 64)
    training.act_fun = "tanh"
    training.ortho_init = True
    training.orth_std = 0.5**2
    training.total_timesteps = 2000000
    training.n_steps = 2048
    training.batch_size = 64
    training.lr = 0.0001
    training.optim = "Adam"
    training.gamma = 0.99
    training.update_freq = 1
    training.polyak = 0.005
    training.adv_steps = False
    training.rb_coef = 0.1
    training.max_grad_norm = -1.0
    training.eps_schedule = "linear"  # Scheduler epsilon: constant, linear, cycle
    training.eps = 0.15  # with constant scheduler
    training.start_val = 0  # linear or cycle
    training.end_val = 0.15  # linear or cycle
    training.start_t = 0  # linear or cycle
    training.end_t = 0.75  # linear or cycle
    training.max_val = 0.15  # cycle
    training.max_t = 0.75  # cycle
    training.relax = ConfigDict(
        {
            "beta_schedule": "constant",
            "beta": 1.0,
            "start_val": 0.0,
            "end_val": 1.0,
            "start_t": 0.0,
            "end_t": 0.75,
            "max_val": 0.15,
            "max_t": 0.75,
        }
    )
    training.seed = 0
    training.log_every = 10
    training.checkpoint = False

    if config.train:
        config.training = training

    # Logging
    config.logger = "tensorboard"  # Options:  "aim", or "tensorboard"
    config.project_name = "advrl"
    config.repo = ".aim"  # for aim
    config.experiment = "sarsa"  # for aim
    config.aim_tag = "v1"
    return config
