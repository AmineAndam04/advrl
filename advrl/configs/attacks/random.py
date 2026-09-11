from ml_collections import ConfigDict


def get_config(dist: str):
    config = ConfigDict()
    config.attack_name = "random"
    # Attack params
    distributions = {
        "uniform": ConfigDict({"dist": "uniform", "signed": False}),
        "gaussian": ConfigDict(
            {
                "dist": "gaussian",
                "type": "normal",  # "type" can be multivariate, normal
            }
        ),
        "fixed": ConfigDict(
            {
                "dist": "fixed",
                "type": "normal",  # "type " can be normal, uniform, all_positive, all_negatives
            }
        ),
        "discrete": ConfigDict({"dist": "discrete"}),
    }
    config.eps = 0.15
    config.deterministic = True
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
    config.experiment = "random"  # for aim
    config.aim_tag = "v1"
    distributions[dist].update(config)
    return distributions[dist]
