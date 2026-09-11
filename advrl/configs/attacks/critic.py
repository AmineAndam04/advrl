from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    config.attack_name = "critic"
    # Attack params
    config.num_iter = 10  # PGD iterations
    config.step_size = -1.0  # PGD's step size , if step_size<0 we use step_size=eps/num_iter
    config.eps = 0.15
    config.deterministic = True  # for the agent
    # Evaluation params
    config.num_envs = 10
    config.num_episodes = 50
    config.num_seeds = 20
    config.auto_seeds = True  # automatically generate random seeds
    config.reprd_auto_seeds = 0  # reproducible random seeds with auto_seeds
    config.pre_seeds = ""  # path to file with pre-selected seeds, supported files: json
    config.device = "cpu"
    # Logging
    config.logger = ""  # Options:  "aim", or ""
    config.repo = ".aim"  # for aim
    config.experiment = "critic"  # for aim
    config.aim_tag = "v1"

    return config
