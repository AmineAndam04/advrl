from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    config.attack_name = "mad"
    # Attack params
    config.eps = 0.15
    config.solver = "sgld"  # sgld or pi
    config.sgld = ConfigDict({"beta": 1.0, "step_size": -1.0, "num_iter": 10})
    config.pi = ConfigDict({"num_iter": 1, "xi": 0.000001})
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
    config.experiment = "mad"  # for aim
    config.aim_tag = "v1"
    return config
