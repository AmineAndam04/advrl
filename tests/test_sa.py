from advrl.trainers import TRAINER_REGESTRY
from pathlib import Path


def test_sa_ppo_convex_relax(tmp_path):
    from advrl.configs.sa_ppo import get_config

    config = get_config()
    config.solver = "relax"
    config.exp_name = "pytest"
    config.total_timesteps = 1000
    config.n_steps = 50
    config.eval_every = 500
    trainer = TRAINER_REGESTRY[config.name]
    model = trainer(config=config, workdir=tmp_path)
    model.train()
    assert (Path(model.path) / "config.json").exists()
    assert (Path(model.path) / "command.txt").exists()
    assert (Path(model.path) / "envparams.npz").exists()
    assert (Path(model.path) / "wrappers.json").exists()
    assert (Path(model.path) / "policy.pt").exists()
    assert (Path(model.path) / "critic.pt").exists()


def test_sa_ppo_beta_scheduler(tmp_path):
    from advrl.configs.sa_ppo import get_config

    config = get_config()
    config.solver = "relax"
    config.relax.beta_schedule = "constant"
    config.exp_name = "pytest"
    config.total_timesteps = 1000
    config.n_steps = 50
    config.eval_every = 500
    trainer = TRAINER_REGESTRY[config.name]
    model = trainer(config=config, workdir=tmp_path)
    model.train()


def test_sa_ppo_sgld(tmp_path):
    from advrl.configs.sa_ppo import get_config

    config = get_config()
    config.solver = "sgld"
    config.exp_name = "pytest"
    config.total_timesteps = 1000
    config.n_steps = 50
    config.eval_every = 500
    trainer = TRAINER_REGESTRY[config.name]
    model = trainer(config=config, workdir=tmp_path)
    model.train()


def test_sa_ppo_pi(tmp_path):
    from advrl.configs.sa_ppo import get_config

    config = get_config()
    config.solver = "pi"
    config.exp_name = "pytest"
    config.total_timesteps = 1000
    config.n_steps = 50
    config.eval_every = 500
    trainer = TRAINER_REGESTRY[config.name]
    model = trainer(config=config, workdir=tmp_path)
    model.train()
