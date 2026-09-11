import random
import numpy as np
import torch
import json
import scipy
from functools import partial


def get_metrics(x: np.ndarray):
    metrics = {
        "iqm": scipy.stats.trim_mean(x, proportiontocut=0.25).item(),
        "iqr": scipy.stats.iqr(x).item(),
        "mean": np.mean(x).item(),
        "median": np.median(x).item(),
        "std": np.std(x).item(),
        "min_": np.min(x).item(),
        "max_": np.max(x).item(),
    }
    return metrics


def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """
    Seed the different random generators.

    :param seed:
    :param using_cuda:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def check_env_and_agent(path, env_id, agent):
    with open(f"{path}/hparams.json", "r") as f:
        hparams = json.load(f)
    assert env_id == hparams.get("env_id")
    assert agent.__class__.__name__ == hparams.get("agent")


def set_device(device_str: int):
    if device_str == "auto":
        if torch.cuda.is_available():
            print("GPU is used")
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("MPS is used")
            return torch.device("mps")
        else:
            print("CPU is used")
            return torch.device("cpu")
    else:
        return torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")


def CycleScheduler(start_val, max_val, end_val, start_t, max_t, end_t, current_t):
    if current_t < start_t:
        return start_val
    elif start_t <= current_t and current_t < max_t:
        slope = (max_val - start_val) / (max_t - start_t)
        base = start_val - slope * start_t
        return min(base + slope * current_t, max_val)
    elif max_t <= current_t and current_t < end_t:
        slope = (max_val - end_val) / (max_t - end_t)
        base = end_val - slope * end_t
        return max(base + slope * current_t, end_val)
    else:
        return end_val


def ConstantScheduler(value, current_t):
    return value


def LinearScheduler(start_val, end_val, start_t, end_t, current_t):
    if current_t < start_t:
        return start_val
    elif start_t <= current_t and current_t < end_t:
        slope = (end_val - start_val) / (end_t - start_t)
        base = start_val - slope * start_t
        return min(base + slope * current_t, end_val)
    else:
        return end_val


def set_scheduler(config, var_to_scheduler="eps"):
    if var_to_scheduler == "frac":
        scheduler_type = config.frac_schedule
        value = config.adv_frac
    elif var_to_scheduler == "beta":
        scheduler_type = config.beta_schedule
        value = config.beta
    elif var_to_scheduler == "eps":
        scheduler_type = config.eps_schedule
        value = config.eps
    else:
        raise ValueError(
            f"A scheduler for {var_to_scheduler} is not implemented yet. You can add it to utils/misc.py"
        )
    if scheduler_type == "constant":
        return partial(ConstantScheduler, value=value)
    elif scheduler_type == "linear":
        return partial(
            LinearScheduler,
            start_val=config.start_val,
            end_val=config.end_val,
            start_t=config.start_t,
            end_t=config.end_t,
        )
    elif scheduler_type == "cycle":
        return partial(
            CycleScheduler,
            start_val=config.start_val,
            max_val=config.max_val,
            end_val=config.end_val,
            start_t=config.start_t,
            max_t=config.max_t,
            end_t=config.end_t,
        )
    else:
        raise ValueError("Schedular not implemented ")


def grad_norm(loss, net):
    grads = torch.autograd.grad(
        loss,
        net.parameters(),
        retain_graph=True,
        allow_unused=True,
    )
    norm = torch.tensor(0.0, device=loss.device)
    for g in grads:
        if g is not None:
            norm += g.pow(2).sum()
    return norm.sqrt()
