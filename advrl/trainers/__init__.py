from .atla_ppo import ATLA_PPO
from .cpo_mono_ppo import CPO_Mono_PPO
from .cpo_radial_ppo import CPO_Radial_PPO
from .cpo_sa_ppo import CPO_SA_PPO
from .paad_ppo import PAAD_PPO
from .ppo import PPO
from .ppo_mujoco_advis_is import PPO_MuJoCo_IS
from .ppo_mujoco_advis_pd import PPO_MuJoCo_PDIS
from .ppo_mujoco_advis_wis import PPO_MuJoCo_WIS
from .ppo_mujoco_advis_wpd import PPO_MuJoCo_WPDIS
from .radial_ppo import Radial_PPO
from .radial_sa_ppo import Radial_SA_PPO
from .sa_ppo import SA_PPO
from .wocar_ppo import WocaR_PPO
from .wocar_sa_ppo import WocaR_SA_PPO

TRAINER_REGESTRY = {
    "ppo": PPO,
    "sa_ppo": SA_PPO,
    "cpo_mono_ppo": CPO_Mono_PPO,
    "cpo_sa_ppo": CPO_SA_PPO,
    "cpo_radial_ppo": CPO_Radial_PPO,
    "radial_ppo": Radial_PPO,
    "radial_sa_ppo": Radial_SA_PPO,
    "atla_ppo": ATLA_PPO,
    "paad_ppo": PAAD_PPO,
    "wocar_ppo": WocaR_PPO,
    "wocar_sa_ppo": WocaR_SA_PPO,
    "mujoco_advis_wpd_ppo_mlp": PPO_MuJoCo_WPDIS,
    "mujoco_advis_is_ppo_mlp": PPO_MuJoCo_IS,
    "mujoco_advis_wis_ppo_mlp": PPO_MuJoCo_WIS,
    "mujoco_advis_pd_ppo_mlp": PPO_MuJoCo_PDIS,
}
