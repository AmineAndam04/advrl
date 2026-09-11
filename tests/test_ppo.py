from advrl.trainers import TRAINER_REGESTRY
from pathlib import Path


def test_ppo_mlp_runs(tmp_path):
    from advrl.configs.ppo import get_config

    config = get_config()

    config.exp_name = "pytest"
    config.total_timesteps = 1000
    config.n_steps = 20
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


# def test_ppo_lstm_runs(tmp_path):
#     from advrl.configs.ppo_lstm import get_config

#     config = get_config()

#     config.exp_name = "pytest"
#     config.total_timesteps = 1000
#     config.n_steps = 20
#     config.eval_every = 500

#     trainer = TRAINER_REGESTRY[config.name]

#     model = trainer(config=config, workdir=tmp_path)
#     model.train()
#     assert (Path(model.path) / "config.json").exists()
#     assert (Path(model.path) / "command.txt").exists()
#     assert (Path(model.path) / "envparams.npz").exists()
#     assert (Path(model.path) / "wrappers.json").exists()
#     assert (Path(model.path) / "policy.pt").exists()
#     assert (Path(model.path) / "critic.pt").exists()
