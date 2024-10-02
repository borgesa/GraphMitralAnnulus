"""Dataclasses containing model setting definitions."""

from dataclasses import dataclass, field


@dataclass
class GraphModelSettings:
    """Settings for graph model layers.

    n_encoder_f_maps: Number of feature maps (input dimension)
    n_graph_features: Number of graph features from graph shape
    num_gcn_in_block: Number of GCN layers
    hidden_dim: Dimension of hidden layer in GCN.
    """

    n_encoder_f_maps: int = 8
    n_graph_features: int = 0
    num_gcn_in_block: int = 4
    hidden_dim: int = 16


@dataclass
class ModelSettings:
    """
    n_points_3d_grid: Number of points (each dimension) to sample.
    """

    # Graph model parameter:
    g_model: GraphModelSettings = field(default_factory=GraphModelSettings)

    num_local_layers: int = 3
