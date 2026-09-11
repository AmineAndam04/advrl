import logging
import sys
import os
from typing import Any, Dict
from ml_collections import ConfigDict


class Logger:
    def __init__(
        self,
        mode: str = "tensorboard",
        logger_name: str = "advrl",
        wand_project: str = "advrl",
        run_name: str = "run_01",
        log_dir: str = "runs",
        config: ConfigDict = ConfigDict(),
        auto_init=True,
    ):
        self.mode = mode.lower()
        self.logger_name = logger_name
        self.wand_project = wand_project
        self.run_name = run_name
        self.log_dir = log_dir
        self.config = config

        self.console_logger = logging.getLogger(logger_name)
        self.console_logger.propagate = False
        self.console_logger.setLevel(logging.INFO)
        if not self.console_logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter("[%(levelname)s %(asctime)s] %(name)s %(message)s", "%H:%M:%S")
            )
            self.console_logger.addHandler(handler)
        self.initialized = False
        if auto_init:
            self.initialize_logger()

    def initialize_logger(self):
        if self.mode == "tensorboard":
            from tensorboardX import SummaryWriter

            self.tb_writer = SummaryWriter(log_dir=os.path.join(self.log_dir, self.run_name))
            self.tb_writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s"
                % ("\n".join([f"|{key}|{value}|" for key, value in self.config.to_dict().items()])),
            )
            self.info("Initialized TensorBoard.")
            self.initialized = True

        elif self.mode == "wandb":
            import wandb

            self.wandb_run = wandb.init(
                project=self.wand_project, name=self.run_name, config=self.config.to_dict()
            )
            self.info("Initialized Weights & Biases.")
            self.initialized = True

        elif self.mode == "aim":
            import aim

            self.aim_run = aim.Run(repo=self.config.repo, experiment=self.config.experiment)
            if self.config.aim_tag != "":
                self.aim_run.add_tag(self.config.aim_tag)
            self.aim_run["hparams"] = self.config.to_dict()
            self.info("Initialized Aim.")
            self.initialized = True

        else:
            self.info(f"Running in CONSOLE ONLY mode (or {self.mode} library not found).")

    def info(self, msg: str):
        self.console_logger.info(msg)

    def warn(self, msg: str):
        self.console_logger.warning(msg)

    # def add_tag_to_aim(self, tag):
    #     if self.mode == "aim":
    #         self.aim_run.add_tag(tag)

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        if self.mode == "tensorboard":
            for k, v in metrics.items():
                self.tb_writer.add_scalar(k, v, step)

        elif self.mode == "wandb":
            self.wandb_run.log(metrics, step=step)

        elif self.mode == "aim":
            for k, v in metrics.items():
                self.aim_run.track(v, name=k, step=step)

    def close(self):
        if self.mode == "tensorboard" and hasattr(self, "tb_writer"):
            self.tb_writer.close()
        if self.mode == "wandb" and hasattr(self, "wandb_run"):
            self.wandb_run.finish()
        self.info("Logger closed.")
