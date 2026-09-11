import math

from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    config.name = "cpo_sa_ppo"  # config id
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
    config.solver = "relax"  # sgld or "relax", or "pi"
    config.sgld = ConfigDict({"beta": 1.0, "step_size": -1.0, "num_iter": 10})
    config.relax = ConfigDict(
        {
            "beta_schedule": "linear",
            "beta": 1.0,
            "start_val": 0.0,
            "end_val": 1.0,
            "start_t": 0.0,
            "end_t": 0.75,
            "max_val": 0.15,
            "max_t": 0.75,
        }
    )
    config.pi = ConfigDict({"num_iter": 1, "xi": 0.000001})
    config.eps_schedule = "linear"  # Scheduler epsilon: constant, linear, cycle
    config.eps = 0.15  # with constant scheduler
    config.start_val = 0.0  # linear or cycle
    config.end_val = 0.15  # linear or cycle
    config.start_t = 0.0  # linear or cycle
    config.end_t = 0.75  # linear or cycle
    config.max_val = 0.15  # cycle
    config.max_t = 0.75  # cycle
    config.balance_grad = False
    config.use_pid = False
    config.tolerance = 0.2
    config.lag_init = 0.0
    config.lag_lr = 0.00025
    config.lag_max = 1.0
    config.ema = 0.95
    config.max_deque = 5
    config.kp = 0.0
    config.ki = 0.0
    config.kd = 0.0

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
    config.algo_name = "cpo_sa"
    config.prefix_run_name = "CPO-SA-PPO"
    # Evaluation
    config.num_eval_envs = 1
    config.num_eval_episodes = 10
    config.eval_every = 200  # evaluate every 'eval_every' episode
    config.deterministic = True
    # Save
    config.checkpoint = True
    config.lock()
    return config
