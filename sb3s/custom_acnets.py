import gym
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy

from sb3s.gnn import GNN
from utils.tools import *


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        config: dict,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        if len(config.policy_net.dims) > 0:
            self.latent_dim_pi = config.policy_net.dims[-1]
        else:
            self.latent_dim_pi = feature_dim
        if len(config.value_net.dims) > 0:
            self.latent_dim_vf = config.value_net.dims[-1]
        else:
            self.latent_dim_vf = feature_dim

        # Shared network
        in_dim = feature_dim
        self.shared_net = []
        for dim, act in zip(config.shared_net.dims, config.shared_net.acts):
            self.shared_net.append(nn.Linear(in_dim, dim))
            if act == "relu":
                self.shared_net.append(nn.ReLU())
            elif act == "tanh":
                self.shared_net.append(nn.Tanh())
            else:
                raise ValueError(f"{act} is not implemented")
            in_dim = dim
        self.shared_net = nn.Sequential(*self.shared_net)

        # Policy network
        if len(config.shared_net.dims) > 0:
            in_dim = config.shared_net.dims[-1]
        else:
            in_dim = feature_dim
        self.policy_net = []
        for dim, act in zip(config.policy_net.dims, config.policy_net.acts):
            self.policy_net.append(nn.Linear(in_dim, dim))
            if act == "relu":
                self.policy_net.append(nn.ReLU())
            elif act == "tanh":
                self.policy_net.append(nn.Tanh())
            else:
                raise ValueError(f"{act} is not implemented")
            in_dim = dim
        self.policy_net = nn.Sequential(*self.policy_net)

        # Policy network
        if len(config.shared_net.dims) > 0:
            in_dim = config.shared_net.dims[-1]
        else:
            in_dim = feature_dim
        self.value_net = []
        for dim, act in zip(config.value_net.dims, config.value_net.acts):
            self.value_net.append(nn.Linear(in_dim, dim))
            if act == "relu":
                self.value_net.append(nn.ReLU())
            elif act == "tanh":
                self.value_net.append(nn.Tanh())
            else:
                raise ValueError(f"{act} is not implemented")
            in_dim = dim
        self.value_net = nn.Sequential(*self.value_net)

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :return: (Tensor, Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        features = self.shared_net(features)
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: Tensor) -> Tensor:
        return self.policy_net(self.shared_net(features))

    def forward_critic(self, features: Tensor) -> Tensor:
        return self.value_net(self.shared_net(features))


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        config=None,
        *args,
        **kwargs,
    ):
        # configuration for mlp extractor
        self._config = config

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = config.sb3_acnet.ortho_init

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim, self._config.sb3_acnet)


class GNNExtractor(nn.Module):
    def __init__(
        self,
        num_slots: int,
        slot_size: int,
        sb3_acnet_config: dict,
    ):
        super(GNNExtractor, self).__init__()
        self.latent_dim_pi = slot_size
        self.latent_dim_vf = slot_size

        self.policy_gnn = GNN(
            input_dim=slot_size,
            hidden_dim=sb3_acnet_config.hidden_dim,
            action_dim=0,
            num_objects=num_slots,
            ignore_action=True,
            copy_action=False,
            act_fn=sb3_acnet_config.act_fn,
            layer_norm=sb3_acnet_config.layer_norm,
            num_layers=sb3_acnet_config.num_layers,
            use_interactions=sb3_acnet_config.use_interactions,
            edge_actions=False,
            output_dim=None)

        self.value_gnn = GNN(
            input_dim=slot_size,
            hidden_dim=sb3_acnet_config.hidden_dim,
            action_dim=0,
            num_objects=num_slots,
            ignore_action=True,
            copy_action=False,
            act_fn=sb3_acnet_config.act_fn,
            layer_norm=sb3_acnet_config.layer_norm,
            num_layers=sb3_acnet_config.num_layers,
            use_interactions=sb3_acnet_config.use_interactions,
            edge_actions=False,
            output_dim=None)

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        return (self.policy_gnn(features, action=None)[0].mean(dim=1),
                self.value_gnn(features, action=None)[0].mean(dim=1))

    def forward_actor(self, features: Tensor) -> Tensor:
        return self.policy_gnn(features, action=None)[0].mean(dim=1)

    def forward_critic(self, features: Tensor) -> Tensor:
        return self.value_gnn(features, action=None)[0].mean(dim=1)


class GNNActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = None,
        config=None,
        *args,
        **kwargs,
    ):
        # configuration for mlp extractor
        self._config = config

        super(GNNActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = config.sb3_acnet.ortho_init

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = GNNExtractor(
            self._config.ocr.slotattr.num_slots, self._config.ocr.slotattr.slot_size, self._config.sb3_acnet
        )
