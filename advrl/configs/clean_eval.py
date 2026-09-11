from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    config.num_envs = 10
    config.num_episodes = 50
    config.num_seeds = 20
    config.auto_seeds = True  # automatically generate random seeds
    config.reprd_auto_seeds = -1  # reproducible random seeds with auto_seeds
    config.pre_seeds = ""  # path to file with pre-selected seeds, supported files: json
    config.deterministic = True
    config.device = "cpu"

    config.logger = ""  # Options:  "aim", or ""
    config.repo = ".aim"  # for aim
    config.experiment = "clean"  # for aim
    config.aim_tag = "v1"
    return config
