import torch
from .base_eval import Base_Eval


class Clean_Eval(Base_Eval):
    def __init__(self, path, config, workdir="evals", command=""):
        super().__init__(path, config, workdir, command)
        self._setup(type_eval="clean")

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
            with torch.no_grad():
                action, h = self.policy.get_action(
                    obs, h=h, deterministic=self.config.deterministic
                )
                action = action.reshape(
                    (self.config.num_envs,) + self.env.single_action_space.shape
                )
            obs, _, _, _, infos = self.env.step(action.cpu().numpy())
            for i, info in enumerate(infos):
                if "episode" in info:
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
