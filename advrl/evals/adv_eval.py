import torch
from advrl.attacks import ATTACK_REGISTRY
from .base_eval import Base_Eval
from ml_collections import ConfigDict


class Adv_Eval(Base_Eval):
    def __init__(self, path, config, workdir="evals", command=""):
        super().__init__(path, config, workdir, command)
        self._setup()

    def _setup(self):
        if self.config.attack_name == "random":
            type_eval = f"{self.config.attack_name}/{self.config.dist}"
        else:
            type_eval = self.config.attack_name
        super()._setup(type_eval=type_eval)
        # Get tge adversary
        agent = self.critic if self.config.attack_name == "critic" else self.policy
        self.adversary = ATTACK_REGISTRY[self.config.attack_name](
            config=self.config, agent=agent, low=self.low, high=self.high
        )
        if self.config.attack_name in ("sarsa", "atla", "paad"):
            # training config
            cfg_train = self.config.training
            cfg_train.env_id = self.agent_config.env_id
            # Set up logging for training
            self.logger.mode = self.config.logger
            self.logger.project_name = self.config.project_name
            self.logger.log_dir = self.save_path
            config = self.config.to_dict()
            config["env_id"] = self.agent_config.env_id
            config["agent"] = self.exp_name
            self.logger.config = ConfigDict(config)
            self.logger.info("Start training the attack")
            self.adversary.train(
                cfg_train=cfg_train,
                wrappers=self.wrappers,
                envparams=self.envparams,
                workdir=self.workdir,
                save_path=self.save_path,
                logger=self.logger,
            )

    def evaluate(self):
        results = {
            f"seed_{seed}": {"seed": seed, "rewards": [], "episode_lengths": [], "num_episodes": 0}
            for seed in self.seeds
        }
        obs, _ = self.env.reset()
        h = None
        current_seed_idx = self.config.num_envs - 1
        running_seeds = self.seeds[: self.config.num_envs]
        completed_seeds = 0
        while completed_seeds < len(self.seeds):
            obs = torch.Tensor(obs).to(self.device)
            #! START: adv
            adv_obs = self.adversary.get_perturbation(obs=obs, h=h)
            assert adv_obs.shape == obs.shape
            #! END: adv
            with torch.no_grad():
                action, h = self.policy.get_action(
                    obs=adv_obs, h=h, deterministic=self.config.deterministic
                )
                action = action.reshape(
                    (self.config.num_envs,) + self.env.single_action_space.shape
                )
            obs, _, _, _, infos = self.env.step(action.cpu().numpy())
            for i, info in enumerate(infos):
                if "episode" in info:
                    self.adversary.reset_hidden(index=i)
                    if h is not None:
                        if isinstance(h, tuple):
                            h[0][:, i] = torch.zeros_like(h[0][:, i])
                            h[1][:, i] = torch.zeros_like(h[1][:, i])
                    results_key = f"seed_{running_seeds[i]}"
                    if results[results_key]["num_episodes"] < self.config.num_episodes:
                        ep_rd = info["episode"]["r"]
                        ep_len = info["episode"]["l"]
                        results[results_key]["rewards"].append(ep_rd)
                        results[results_key]["episode_lengths"].append(ep_len)
                        results[results_key]["num_episodes"] = (
                            results[results_key]["num_episodes"] + 1
                        )
                        if results[results_key]["num_episodes"] == self.config.num_episodes:
                            completed_seeds += 1
                            # enough for this seed, replace it with the next seed
                            current_seed_idx += 1
                            if current_seed_idx < len(self.seeds):
                                new_seed = self.seeds[current_seed_idx]
                                running_seeds[i] = new_seed
                                obs_i, _ = self.env.reset_ith_env(env_idx=i, seed=new_seed)
                                obs[i] = obs_i

        # check the results
        self._check_results(results)
        stats = self.compute_stats(results)
        # Log the results
        self.logger.info(stats)
        # save
        self.save(stats, results)

        # close
        self.env.close_extras()
        self.logger.close()
