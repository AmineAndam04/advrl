# adv-RL

Adversarial Reinforcement Learning

# How to use

We use `uv` as package manager. Visit [their webpage](https://docs.astral.sh/uv/) to install it.

**Step 1: setup the uv environemnt**

```
# Create uv environment
cd adv-RL
uv venv
source .venv/bin/activate
# Install dependencies
uv pip install -r requirements.txt
```

**Step 2: lunch the training**

Each algorithm has its own config file, check `advrl/configs` for the list of parameters.

Example to train with PAAD

```
python run.py --config=advrl/configs/paad_ppo.py \
    --config.env_id="HalfCheetah-v5" --config.total_timesteps=10000 \
    --config.lr_actor=0.00005 --config.lr_critic=0.00005 \
    --config.paad.lr_actor=0.0003 --config.paad.lr_critic=0.0001 
```

Example to train Advis with CPO-Radial:

```
python run.py --config=advrl/configs/cpo_radial_ppo.py \
    --config.exp_name="v1" \
    --config.env_id="HalfCheetah-v5" \
    --config.log_std_init=-1 \
    --config.end_val=0.075 --config.eps=0.075 \
    --config.lr_actor=0.0004 \
    --config.lr_critic=0.0025 \
    --config.ent_coef=0.001 \
    --config.lag_init=0 \
    --config.lag_lr=0.001 \
    --config.tolerance=0.5 \
    --config.max_grad_norm=-1 \
    --config.num_eval_envs=10 \
    --config.eval_every=600 \
    --config.total_timesteps=2000000 \
    --config.relax.beta_schedule='constant' \
    --config.eps_schedule='constant' 
```

After training the agent, a checkpoint will be saved at `workdir/env_id/name_algo/logging_id`.

**Step 3: run evaluations**

To evaluate the clean performance of a trained agent use:

```
python mujoco_evaluate.py --config=advrl/configs/clean_eval.py --agent=put_path_to_agent  --config.num_envs=1 --config.num_episodes=1 --config.num_seeds=1
```

To evaluate against adversarial attackers, use:

1. Random attacks:
You can use other random attakcs, check the available options at `adv-RL/advrl/configs/attacks/random.py`

```
python mujoco_attack.py --config=advrl/configs/attacks/random.py:uniform --agent=path_to_ur_agent --config.num_envs=1 --config.num_episodes=1 --config.num_seeds=1 --config.eps=0.15 
```

2. Critic:

```
python mujoco_attack.py --config=advrl/configs/attacks/critic.py --agent=path_to_ur_agent --config.num_envs=1 --config.num_episodes=1 --config.num_seeds=1 --config.eps=0.15 
```

3. MAD

```
python mujoco_attack.py --config=advrl/configs/attacks/mad.py --agent=path_to_ur_agent --config.num_envs=1 --config.num_episodes=1 --config.num_seeds=1 --config.eps=0.15 
```

4. Robust SARSA:
Training hyper-paramters can be overwritten using `--config.training`. Check `advrl/configs/attacks/sarsa.py` for the full list of hyperparameters.

```
python mujoco_attack.py --config=advrl/configs/attacks/sarsa.py --agent=path_to_ur_agent --config.num_envs=1 --config.num_episodes=1 --config.num_seeds=1 --config.eps=0.15 --config.training.lr=0.001 --config.training.total_timesteps=1000
```

5. SA-RL

```
python mujoco_attack.py --config=advrl/configs/attacks/atla.py --agent=path_to_ur_agent --config.num_envs=1 --config.num_episodes=1 --config.num_seeds=1 --config.eps=0.15 --config.training.lr_actor=0.001 --config.training.total_timesteps=5000
```

6. PA-AD

```
python mujoco_attack.py --config=advrl/configs/attacks/paad.py --agent=path_to_ur_agent --config.num_envs=1 --config.num_episodes=1 --config.num_seeds=1 --config.eps=0.15 --config.training.lr_actor=0.001 --config.training.total_timesteps=5000
```

**Parallel evaluations:**
You can run parallel evaluations using the available CPU cores or allocated CPUs by a slurm job when evaluating on HPC. Use the `--parallel` flag to enable parallel evaluations.

```
python eval_all.py --agent=path_to_ur_agent --attack=random --epsilon=0.15 --parallel
```
