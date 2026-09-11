from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    config.attack_name = "bound"
    # Attack params
    config.eps = 0.15
    config.deterministic = True
    # Evaluation params
    config.num_envs = 10
    config.num_episodes = 20
    config.num_seeds = 10
    config.auto_seeds = True  # automatically generate random seeds
    config.reprd_auto_seeds = 0  # reproducible random seeds with auto_seeds
    config.pre_seeds = ""  # path to file with pre-selected seeds, supported files: json
    config.device = "cpu"
    # Logging
    config.logger = "aim"  # Options:  "aim", or ""
    config.repo = ".aim"  # for aim
    config.experiment = "mad"  # for aim
    config.aim_tag = "v1"
    return config
