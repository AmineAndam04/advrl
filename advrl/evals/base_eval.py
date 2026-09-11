from abc import ABC, abstractmethod
import os
import uuid
import json
import random
import datetime
import numpy as np
from advrl.utils.save import load_saved_folder
from advrl.envs.vec_env import VecEnv
from advrl.envs.mujoco import create_make_env_mujoco_from_json
from advrl.utils.misc import set_device, get_metrics
from advrl.utils.logger import Logger
from advrl.utils.envs import env_bounds
from advrl.agents import NETWORK_REGISTRY
from gymnasium.spaces import flatdim
from ml_collections import ConfigDict


class Base_Eval(ABC):
    def __init__(self, path, config, workdir="evals", command=""):
        self.path = path
        self.config = config
        self.workdir = workdir
        self.command = command

    def _setup(self, type_eval="clean"):
        """
        Check the eval config and prepare the environment and seeds
        """
        self.logger = Logger(auto_init=False, mode=self.config.logger)
        if self.config.auto_seeds:
            self.logger.info("Automatically generating random seeds")
            if self.config.reprd_auto_seeds >= 0:
                random.seed(self.config.reprd_auto_seeds)
            self.seeds = random.sample(range(0, 2**32), self.config.num_seeds)
        else:
            assert self.config.pre_seeds != ""
            self.logger.info("Fetching pre-defined seeds")
            with open(self.config.pre_seeds) as f:
                self.seeds = json.load(f)
        self.device = set_device(self.config.device)
        # Import the saved files
        self.agent_config, self.wrappers, self.envparams, self.state_dicts = load_saved_folder(
            self.path
        )
        # Load the environment
        assert self.config.num_envs <= len(self.seeds)
        self._load_env()
        self.logger.info("Successfully Loaded the environment")
        # Load the agent
        self._load_agent()
        self.logger.info("Successfully Loaded the agent")
        self.time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.exp_name = self.path.strip("/").split("/")[-1]
        self.exp_type = self.path.strip("/").split("/")[-2]
        run_id = uuid.uuid4().hex
        self.save_path = f"{self.workdir}/{self.agent_config.env_id}/{self.exp_type}/{self.exp_name}/{type_eval}/{self.time_token}/{run_id}"
        self.logger.info(f"Agent: {self.exp_name}")

    def _load_env(self):
        self.env = VecEnv(
            [
                create_make_env_mujoco_from_json(
                    wrappers=self.wrappers,
                    envparams=self.envparams,
                    env_id=self.agent_config.env_id,
                    seed=self.seeds[i],
                    gamma=self.agent_config.gamma,
                    index=i,
                    update_stats=False,
                )
                for i in range(self.config.num_envs)
            ]
        )
        # Lower and upper bound of observations
        clip_obs = "TransformObservation" in self.wrappers
        if clip_obs:
            range_obs = self.envparams["range_obs"]
        else:
            range_obs = ()
        self.low, self.high = env_bounds(
            env=self.env,
            clip_obs=clip_obs,
            range_obs=range_obs,
        )

    def _load_agent(
        self,
    ):
        if "ppo" in self.agent_config.name:
            if "policy" in self.state_dicts.keys():
                self.policy = NETWORK_REGISTRY[self.agent_config.actor]
                self.policy = self.policy(
                    in_dim=flatdim(self.env.single_observation_space),
                    out_dim=flatdim(self.env.single_action_space),
                    net_arch=ConfigDict(
                        {
                            "pi_net": self.agent_config.pi_net,
                            "act_fun": self.agent_config.pi_act_fun,
                            "pi_lstm_dim": self.agent_config.pi_net,
                            "ortho_init": self.agent_config.ortho_init,
                            "orth_std": self.agent_config.orth_std,
                            "log_std_init": self.agent_config.log_std_init,
                        }
                    ),
                ).to(self.device)
                self.policy.load_state_dict(self.state_dicts["policy"])
                self.policy.eval()
            if "critic" in self.state_dicts.keys():
                self.critic = NETWORK_REGISTRY[self.agent_config.critic]
                self.critic = self.critic(
                    in_dim=flatdim(self.env.single_observation_space),
                    net_arch=ConfigDict(
                        {
                            "vf_net": self.agent_config.vf_net,
                            "act_fun": self.agent_config.vf_act_fun,
                            "vf_lstm_dim": self.agent_config.vf_net[-1],
                            "ortho_init": self.agent_config.ortho_init,
                            "orth_std": self.agent_config.orth_std,
                        }
                    ),
                ).to(self.device)
                self.critic.load_state_dict(self.state_dicts["critic"])
                self.critic.eval()

    # def _unnormalize_obs(self, obs):
    #     return obs * torch.sqrt(self.normalize_vars + 1e-8) + self.normalize_means

    # def _normalize_obs(self, obs):
    #     return (obs - self.normalize_means) / torch.sqrt(self.normalize_vars + 1e-8)

    @abstractmethod
    def evaluate(self):
        pass

    def save(self, stats, results):
        os.makedirs(self.save_path, exist_ok=True)
        with open(f"{self.save_path}/metrics.json", "w") as f:
            json.dump(stats, f, indent=2)
        with open(f"{self.save_path}/metadata.json", "w") as f:
            json.dump(results, f, indent=2)
        with open(f"{self.save_path}/seeds.json", "w") as f:
            json.dump(self.seeds, f, indent=2)
        with open(f"{self.save_path}/config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        with open(f"{self.save_path}/agent.json", "w") as f:
            json.dump(self.agent_config.to_dict(), f, indent=2)
        if self.logger.mode == "aim":
            if self.logger.initialized is False:
                config = self.config.to_dict()
                config["env_id"] = self.agent_config.env_id
                config["agent"] = self.exp_name  # Path(self.path).name
                self.logger.config = ConfigDict(config)
                self.logger.initialize_logger()
            self.logger.log_metrics(metrics=stats["rewards"], step=0)

    def _check_results(self, results):
        for seed in self.seeds:
            results_seed = results[f"seed_{seed}"]
            assert results_seed["seed"] == seed
            assert len(results_seed["rewards"]) == self.config.num_episodes
            assert len(results_seed["episode_lengths"]) == self.config.num_episodes
            assert results_seed["num_episodes"] == self.config.num_episodes

    def compute_stats(self, results):
        rewards = np.array([r for v in results.values() for r in v["rewards"]])
        episode_lengths = np.array([r for v in results.values() for r in v["episode_lengths"]])
        reward_metrics = get_metrics(rewards)
        episode_lengths_metrics = get_metrics(episode_lengths)
        stats = {"rewards": reward_metrics, "episode_lengths": episode_lengths_metrics}
        return stats
