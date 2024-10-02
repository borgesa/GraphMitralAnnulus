#!/usr/bin/env python3
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch_geometric.nn import DeepGCNLayer, DenseGCNConv

from mvq_graph_model.model.settings import GraphModelSettings


@dataclass
class GCNLayerConfig:
    """Dataclass for initializing GCN layers."""

    in_channels: int
    out_channels: int
    conv_type: type[nn.Module] = DenseGCNConv
    activation: nn.Module = nn.ReLU(inplace=False)
    block: str = "res+"


def get_gcn_config(settings: GraphModelSettings) -> list[GCNLayerConfig]:
    """Returns list of configurations for GCN Block.

    The first layer uses 'plain' block in order to go from the input
    dimensionality (number of feature maps) to the hidden dimensionality.

    The remaining layers use the default or specified block.
    """
    hidden_dim = settings.hidden_dim
    # First layer configuration, hard coded to use 'plain' block.
    # Residual type layer requires in-out shape to be the same
    in_features = settings.n_encoder_f_maps + settings.n_graph_features
    gcn_config = [
        GCNLayerConfig(
            in_channels=in_features,
            out_channels=hidden_dim,
            block="plain",
        )
    ]
    # Additional layers configuration:
    gcn_config.extend(
        [
            GCNLayerConfig(settings.hidden_dim, out_channels=hidden_dim)
            for _ in range(settings.num_gcn_in_block - 1)
        ]
    )
    return gcn_config


class GCNBlock(nn.Module):
    """A class for creating a block of GCN layers using settings in GCNLayerConfig.

    Args:
        settings: Instance of GraphModelSettings, for GCN layer settings
        out_dim (int): Output dimension, which varies based on the operation mode.

    TODO: Evaluate the 'mode' part. Discuss w/David
        Not convinced the difference is meaningful (other than output dim)
    """

    def __init__(
        self,
        settings: GraphModelSettings,
        mean_aggregate: bool,
        out_dim: int,
    ):
        super().__init__()

        layer_configs = get_gcn_config(settings=settings)

        self.mean_aggregate = mean_aggregate
        self.out_dim = out_dim

        self.gcn_list = nn.ModuleList()
        for config in layer_configs:
            conv_layer = self.create_gcn_layer(config)
            self.gcn_list.append(conv_layer)

        # Extract the number of filters from the last layer's configuration
        last_layer_out_channels = layer_configs[-1].out_channels

        self.lin = nn.Linear(last_layer_out_channels, out_dim)

    @staticmethod
    def create_gcn_layer(config: GCNLayerConfig) -> DeepGCNLayer:
        """
        Create a GCN layer based on the provided configuration.

        Args:
            config (GCNLayerConfig): Configuration for the GCN layer.

        Return:
            DeepGCNLayer: The created GCN layer.
        """
        # Create and return the GCN layer using the configuration from the data class
        return DeepGCNLayer(
            conv=config.conv_type(config.in_channels, config.out_channels),
            act=config.activation,
            block=config.block,
        )

    def forward(
        self,
        features: torch.Tensor,
        edges: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the GCN layers.

        Args:
            features: Features for each point on the graph.
            edges: Graph edge matrix.

        Return:
            torch.Tensor: The output of the forward pass.
        """
        for gcn_layer in self.gcn_list:
            features = gcn_layer(features, edges)

        if self.mean_aggregate:
            # Could use the below, but likely not w/current implementation:
            # from torch_geometric.nn import global_mean_pool
            features = torch.mean(features, dim=1, keepdim=False)

        return self.lin(input=features)
