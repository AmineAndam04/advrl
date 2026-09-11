from .continuous import ActorMLP, RelaxedActorMLP, CriticMLP, QnetworkContinuousMLP

NETWORK_REGISTRY = {
    "pi_mlp": ActorMLP,
    "vf_mlp": CriticMLP,
    "qnet_continuous_mlp": QnetworkContinuousMLP,
}

RELAXED_NETWORK_REGISTRY = {"pi_mlp": RelaxedActorMLP}
