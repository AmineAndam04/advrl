import torch
import numpy as np


def orth_init(layer, std=0.01, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def no_init(layer, **kwargs):
    return layer


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return min(slope * t + start_e, end_e)
