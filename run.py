import sys
from absl import app
from absl import flags
from ml_collections import config_flags
from advrl.trainers import TRAINER_REGESTRY

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "workdir",
    default="runs",
    help="Directory to save checkpoints and logging info.",
)

config_flags.DEFINE_config_file(
    "config",
    "configs/sa_ppo.py",
    "File path to the default configuration file.",
    lock_config=True,
)


def main(argv):
    config = FLAGS.config
    trainer = TRAINER_REGESTRY[config.name]
    model = trainer(config=config, workdir=FLAGS.workdir, command=sys.argv)
    model.train()


if __name__ == "__main__":
    app.run(main)
