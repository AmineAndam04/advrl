from .random import RandomAttack
from .mad import MAD
from .critic import CriticAttack
from .sarsa import SarsaAttack
from .atla import ATLAAttack
from .paad import PAADAttack

ATTACK_REGISTRY = {
    "random": RandomAttack,
    "mad": MAD,
    "critic": CriticAttack,
    "sarsa": SarsaAttack,
    "atla": ATLAAttack,
    "paad": PAADAttack,
}
