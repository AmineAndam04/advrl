from absl import app
from absl import flags
from itertools import product
import subprocess
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import random

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "workdir",
    default="evals",
    help="Directory to save checkpoints and logging info.",
)
flags.DEFINE_string(
    "agent",
    default="",
    help="Path of the agent to evaluate.",
)
flags.DEFINE_string(
    "attack",
    default="mad",
    help="Name of the attack to run.",
)
flags.DEFINE_bool(
    "lstm",
    default=False,
    help="LSTM nets for ATLA and PAAD",
)
flags.DEFINE_bool(
    "parallel",
    default=False,
    help="Run parallel attacks",
)
flags.DEFINE_integer(
    "seed",
    default=None,
    help="Override training seed",
)
flags.DEFINE_float(
    "epsilon",
    default=0.075,
    help="Attack budget (eps).",
)
flags.DEFINE_integer(
    "num_envs",
    default=10,
    help="Number of environments.",
)
# TODO aim_tag
# All
# NUM_ENVS = 10
NUM_EPISODES = 50
NUM_SEEDS = 20
AUTO_SEEDS = True
REPRD_AUTO_SEEDS = 0
STEP_SIZES = [-1]  #! weights ...
AIM_TAG = "v1"
AIM_REPO = "/home/amine.andam/lustre/vr_outsec-vh2sz1t4fks/users/amine.andam/Rpbustness/.aim"
# ATLA
ATLA_TOTAL_TIMESTEPS = 2000000
# PAAD
PAAD_LOG_STD_INIT = -1
PAAD_TOTAL_TIMESTEPS = 2000000


def get_grids(epsilon):
    attack_grid = {
        "random": {
            "config_file": "advrl/configs/attacks/random.py",
            "grid": {
                "uniform": {
                    "config.signed": [True, False],
                    "config.deterministic": [True],
                    "config.eps": [epsilon],
                },
                "gaussian": {
                    "config.type": ["multivariate", "normal"],
                    "config.deterministic": [True],
                    "config.eps": [epsilon],
                },
                "fixed": {
                    "config.type": ["normal", "uniform", "all_positive", "all_negatives"],
                    "config.deterministic": [True],
                    "config.eps": [epsilon],
                },
                "discrete": {
                    "config.dist": ["discrete"],
                    "config.deterministic": [True],
                    "config.eps": [epsilon],
                },
            },
        },
        "critic": {
            "config_file": "advrl/configs/attacks/critic.py",
            "grid": {
                "config.num_iter": [10],
                "config.step_size": STEP_SIZES,
                "config.eps": [epsilon],
                "config.deterministic": [True],
            },
        },
        "mad": {
            "config_file": "advrl/configs/attacks/mad.py",
            "grid_sgld": {
                "config.solver": ["sgld"],
                "config.eps": [epsilon],
                "config.deterministic": [True],
                "config.sgld.step_size": STEP_SIZES,
                "config.sgld.beta": [1000, 10000, 100000],
            },
            "grid_pi": {
                "config.solver": ["pi"],
                "config.eps": [epsilon],
                "config.deterministic": [True],
                "config.pi.num_iter": [1],
                "config.pi.xi": [0.0001, 0.00001, 0.01],
            },
        },
        "sarsa": {
            "config_file": "advrl/configs/attacks/sarsa.py",
            "grid": {
                "config.training.lr": [0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005],
                "config.training.net_arch": ["(64, 64)"],
                "config.training.act_fun": ["tanh"],
                "config.training.adv_steps": [False],
                "config.training.rb_coef": [0, 0.001, 0.01, 0.1, 0.5, 1],
                "config.training.polyak": [0.001, 0.005, 0.01, 0.05],
                "config.training.n_steps": [2048],
                "config.training.batch_size": [64],
                "config.training.num_envs": [2],
                "config.training.seed": [0],
                "config.step_size": STEP_SIZES,
                "config.eps": [epsilon],
                "config.deterministic": [True],
            },
        },
        "atla": {
            "config_file": "advrl/configs/attacks/atla.py",
            "grid": {
                "config.training.total_timesteps": [ATLA_TOTAL_TIMESTEPS],
                "config.training.squash": [False, True],
                "config.training.log_std_init": [-1, -2],
                "config.training.lr_actor": [0.0001, 0.0005, 0.001, 0.0025, 0.005],
                "config.training.lr_critic": [0.0001, 0.0005, 0.001, 0.0025, 0.005],
                "config.training.n_steps": [2048],
                "config.training.pi_act_fun": ["tanh"],
                "config.training.ent_coef": [0, 0.00001],
                "config.training.seed": [0],
                "config.eps": [epsilon],
                "config.deterministic": [True],
            },
        },
        "paad": {
            "config_file": "advrl/configs/attacks/paad.py",
            "grid": {
                "config.training.total_timesteps": [PAAD_TOTAL_TIMESTEPS],
                "config.training.log_std_init": [PAAD_LOG_STD_INIT],
                "config.training.lr_actor": [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01],
                "config.training.lr_critic": [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01],
                "config.training.n_steps": [2048],
                "config.training.ent_coef": [0, 0.0001, 0.001],
                "config.training.pi_act_fun": ["tanh", "relu"],
                "config.training.clip_range": [0.2, 0.3],
                "config.training.seed": [0],
                "config.training.num_iter": [1],
                "config.eps": [epsilon],
                "config.step_size": STEP_SIZES,
                "config.deterministic": [True],
            },
        },
    }
    return attack_grid


def random_attack_cmd(attack_grid, agent, workdir, num_envs, lstm=None):
    cmds = []
    exps = []
    config_file = attack_grid["random"]["config_file"]
    base_cmd = [
        "python",
        "mujoco_attack.py",
        f"--workdir={workdir}",
        f"--agent={agent}",
    ]
    grid = attack_grid["random"]["grid"]
    rnd_attacks = grid.keys()
    for rnd_attack in rnd_attacks:
        name_hparams = grid[rnd_attack].keys()
        val_hparams = [grid[rnd_attack][var] for var in name_hparams]
        for combination in product(*val_hparams):
            cmd = base_cmd[:]
            exp = ""
            cmd.extend(
                [
                    f"--config={config_file}:{rnd_attack}",
                    f"--config.num_envs={num_envs}",
                    f"--config.num_episodes={NUM_EPISODES}",
                    f"--config.num_seeds={NUM_SEEDS}",
                    f"--config.auto_seeds={AUTO_SEEDS}",
                    f"--config.reprd_auto_seeds={REPRD_AUTO_SEEDS}",
                    f"--config.aim_tag={AIM_TAG}",
                    f"--config.repo={AIM_REPO}",
                ]
            )
            for k, v in zip(name_hparams, combination):
                cmd.append(f"--{k}={v}")
                exp += f"--{k}={v} "
            cmds.append(cmd)
            exps.append(exp)
    return cmds, exps


def mad_attack_cmd(attack_grid, agent, workdir, num_envs, lstm=None):
    cmds = []
    exps = []
    config_file = attack_grid["mad"]["config_file"]
    base_cmd = [
        "python",
        "mujoco_attack.py",
        f"--workdir={workdir}",
        f"--agent={agent}",
        f"--config={config_file}",
        f"--config.num_envs={num_envs}",
        f"--config.num_episodes={NUM_EPISODES}",
        f"--config.num_seeds={NUM_SEEDS}",
        f"--config.auto_seeds={AUTO_SEEDS}",
        f"--config.reprd_auto_seeds={REPRD_AUTO_SEEDS}",
        f"--config.aim_tag={AIM_TAG}",
        f"--config.repo={AIM_REPO}",
    ]
    for grid in [attack_grid["mad"]["grid_sgld"], attack_grid["mad"]["grid_pi"]]:
        name_hparams = grid.keys()
        val_hparams = [grid[var] for var in name_hparams]
        for combination in product(*val_hparams):
            cmd = base_cmd[:]
            exp = ""
            for k, v in zip(name_hparams, combination):
                cmd.append(f"--{k}={v}")
                exp += f"--{k}={v} "
            cmds.append(cmd)
            exps.append(exp)
    return cmds, exps


def critic_attack_cmd(attack_grid, agent, workdir, num_envs, lstm=None):
    cmds = []
    exps = []
    config_file = attack_grid["critic"]["config_file"]
    base_cmd = [
        "python",
        "mujoco_attack.py",
        f"--workdir={workdir}",
        f"--agent={agent}",
        f"--config={config_file}",
        f"--config.num_envs={num_envs}",
        f"--config.num_episodes={NUM_EPISODES}",
        f"--config.num_seeds={NUM_SEEDS}",
        f"--config.auto_seeds={AUTO_SEEDS}",
        f"--config.reprd_auto_seeds={REPRD_AUTO_SEEDS}",
        f"--config.aim_tag={AIM_TAG}",
        f"--config.repo={AIM_REPO}",
    ]
    grid = attack_grid["critic"]["grid"]
    name_hparams = grid.keys()
    val_hparams = [grid[var] for var in name_hparams]
    for combination in product(*val_hparams):
        cmd = base_cmd[:]
        exp = ""
        for k, v in zip(name_hparams, combination):
            cmd.append(f"--{k}={v}")
            exp += f"--{k}={v} "
        cmds.append(cmd)
        exps.append(exp)
    return cmds, exps


def sarsa_attack_cmd(attack_grid, agent, workdir, num_envs, lstm=None):
    cmds = []
    exps = []
    config_file = attack_grid["sarsa"]["config_file"]
    base_cmd = [
        "python",
        "mujoco_attack.py",
        f"--workdir={workdir}",
        f"--agent={agent}",
        f"--config={config_file}",
        "--config.training.eps_schedule=linear",
        f"--config.num_envs={num_envs}",
        f"--config.num_episodes={NUM_EPISODES}",
        f"--config.num_seeds={NUM_SEEDS}",
        f"--config.auto_seeds={AUTO_SEEDS}",
        f"--config.reprd_auto_seeds={REPRD_AUTO_SEEDS}",
        f"--config.aim_tag={AIM_TAG}",
        f"--config.repo={AIM_REPO}",
    ]
    grid = attack_grid["sarsa"]["grid"]
    name_hparams = grid.keys()
    val_hparams = [grid[var] for var in name_hparams]
    for combination in product(*val_hparams):
        combo_dict = dict(zip(name_hparams, combination))
        if combo_dict.get("config.training.adv_steps") is True:
            if combo_dict.get("config.training.rb_coef") != 0:
                continue
            else:
                combo_dict["config.training.total_timesteps"] = 2000000
        else:
            if combo_dict.get("config.training.rb_coef") != 0:
                combo_dict["config.training.total_timesteps"] = 1000000
            else:
                combo_dict["config.training.total_timesteps"] = 200000

        cmd = base_cmd[:]
        exp = ""
        for k, v in combo_dict.items():
            if k == "config.training.seed" and FLAGS.seed is not None:
                v = FLAGS.seed
            cmd.append(f"--{k}={v}")
            exp += f"--{k}={v} "
            if k == "config.eps":
                cmd.append(f"--config.training.end_val={v}")
        cmds.append(cmd)
        exps.append(exp)
    return cmds, exps


def atla_attack_cmd(attack_grid, agent, workdir, num_envs, lstm=False):
    cmds = []
    exps = []
    config_file = attack_grid["atla"]["config_file"]
    base_cmd = [
        "python",
        "mujoco_attack.py",
        f"--workdir={workdir}",
        f"--agent={agent}",
        f"--config={config_file}",
        "--config.training.eps_schedule=constant",
        f"--config.num_envs={num_envs}",
        f"--config.num_episodes={NUM_EPISODES}",
        f"--config.num_seeds={NUM_SEEDS}",
        f"--config.auto_seeds={AUTO_SEEDS}",
        f"--config.reprd_auto_seeds={REPRD_AUTO_SEEDS}",
        f"--config.aim_tag={AIM_TAG}",
        f"--config.repo={AIM_REPO}",
    ]
    grid = attack_grid["atla"]["grid"]
    name_hparams = grid.keys()
    val_hparams = [grid[var] for var in name_hparams]
    for combination in product(*val_hparams):
        cmd = base_cmd[:]
        exp = ""
        for k, v in zip(name_hparams, combination):
            if k == "config.training.seed" and FLAGS.seed is not None:
                v = FLAGS.seed
            cmd.append(f"--{k}={v}")
            exp += f"--{k}={v} "
            if k == "config.eps":
                cmd.append(f"--config.training.end_val={v}")
            if k == "config.training.pi_act_fun":
                cmd.append(f"--config.training.vf_act_fun={v}")
                exp += f"--config.training.vf_act_fun={v} "
        cmds.append(cmd)
        exps.append(exp)
    mlp_config = [
        "--config.training.actor=pi_mlp",
        "--config.training.critic=vf_mlp",
        "--config.training.actor_lstm=False",
        "--config.training.critic_lstm=False",
    ]
    lstm_config = [
        "--config.training.actor=pi_lstm",
        "--config.training.critic=vf_lstm",
        "--config.training.actor_lstm=True",
        "--config.training.critic_lstm=True",
    ]
    if lstm:
        return [cmd + lstm_config for cmd in cmds], exps
    else:
        return [cmd + mlp_config for cmd in cmds], exps


def paad_attack_cmd(attack_grid, agent, workdir, num_envs, lstm=False):
    cmds = []
    exps = []  # which hyper-param combination
    config_file = attack_grid["paad"]["config_file"]
    base_cmd = [
        "python",
        "mujoco_attack.py",
        f"--workdir={workdir}",
        f"--agent={agent}",
        f"--config={config_file}",
        "--config.training.eps_schedule=constant",
        f"--config.num_envs={num_envs}",
        f"--config.num_episodes={NUM_EPISODES}",
        f"--config.num_seeds={NUM_SEEDS}",
        f"--config.auto_seeds={AUTO_SEEDS}",
        f"--config.reprd_auto_seeds={REPRD_AUTO_SEEDS}",
        f"--config.aim_tag={AIM_TAG}",
        f"--config.repo={AIM_REPO}",
    ]
    grid = attack_grid["paad"]["grid"]
    name_hparams = grid.keys()
    val_hparams = [grid[var] for var in name_hparams]
    for combination in product(*val_hparams):
        cmd = base_cmd[:]
        exp = ""
        for k, v in zip(name_hparams, combination):
            if k == "config.training.seed" and FLAGS.seed is not None:
                v = FLAGS.seed
            cmd.append(f"--{k}={v}")
            exp += f"--{k}={v} "
            if k == "config.eps":
                cmd.append(f"--config.training.end_val={v}")
            if k == "config.training.pi_act_fun":
                cmd.append(f"--config.training.vf_act_fun={v}")
                exp += f"--config.training.vf_act_fun={v} "
            if k == "config.training.clip_range":
                cmd.append(f"--config.training.clip_range_vf={v}")
                exp += f"--config.training.clip_range_vf={v}"
            if k == "config.training.num_iter":
                cmd.append(f"--config.num_iter={v}")
        cmds.append(cmd)
        exps.append(exp)
    mlp_config = [
        "--config.training.actor=pi_mlp",
        "--config.training.critic=vf_mlp",
        "--config.training.actor_lstm=False",
        "--config.training.critic_lstm=False",
    ]
    lstm_config = [
        "--config.training.actor=pi_lstm",
        "--config.training.critic=vf_lstm",
        "--config.training.actor_lstm=True",
        "--config.training.critic_lstm=True",
    ]
    if lstm:
        return [cmd + lstm_config for cmd in cmds], exps
    else:
        return [cmd + mlp_config for cmd in cmds], exps


CMD_GENERATOR = {
    "random": random_attack_cmd,
    "critic": critic_attack_cmd,
    "mad": mad_attack_cmd,
    "sarsa": sarsa_attack_cmd,
    "atla": atla_attack_cmd,
    "paad": paad_attack_cmd,
}


def run_one_experiment(exp_cmd_tuple):
    exp, cmd = exp_cmd_tuple
    print("Starting Experiment:", exp.replace("--config.", "").replace("training.", ""), flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Experiment failed: {exp}")
    print("Finished Experiment:", exp.replace("--config.", "").replace("training.", ""), flush=True)
    return True


def main(argv):
    cmd_generator = CMD_GENERATOR[FLAGS.attack]
    attack_grid = get_grids(epsilon=FLAGS.epsilon)
    cmds, exps = cmd_generator(
        attack_grid=attack_grid,
        agent=FLAGS.agent,
        workdir=FLAGS.workdir,
        num_envs=FLAGS.num_envs,
        lstm=FLAGS.lstm,
    )
    paired = list(zip(exps, cmds))
    random.Random(42).shuffle(paired)
    exps, cmds = zip(*paired)
    print(f"Total experiments: {len(cmds)}", flush=True)
    if FLAGS.parallel:
        # max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        if "SLURM_CPUS_PER_TASK" in os.environ:
            max_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
        else:
            max_workers = os.cpu_count()
        print("Running with", max_workers, "parallel workers", flush=True)
        if max_workers > 1:
            experiments = list(zip(exps, cmds))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(run_one_experiment, exp_cmd) for exp_cmd in experiments]
                for future in as_completed(futures):
                    future.result()
        else:
            raise RuntimeError("Parallel requested but max_workers <= 1")
    else:
        for exp, cmd in zip(exps, cmds):
            print("Experiment: ", exp.replace("--config.", "").replace("training.", ""), flush=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    app.run(main)
