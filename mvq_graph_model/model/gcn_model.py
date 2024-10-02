import torch
import torch.nn as nn
from einops import repeat

from mvq_graph_model.graph.features import sample_coords_from_encoder_features
from mvq_graph_model.model.gcn_block import GCNBlock
from mvq_graph_model.model.global_grid import create_uniform_3d_grid
from mvq_graph_model.model.settings import GraphModelSettings
from mvq_graph_model.utils.geometry import apply_transform, get_affine_matrix


def get_initial_edges(n_points: int) -> nn.Parameter:
    """Graph edges, effectively an adjacency matrix."""
    return nn.Parameter(
        torch.ones(1, n_points, n_points) / n_points, requires_grad=True
    )


class GlobalGraphLayer(nn.Module):
    def __init__(
        self,
        settings: GraphModelSettings,
        points_per_dim: int,
        out_dim_graph: int,
        shape_template: torch.Tensor,
        detached_feature_map: bool = False,
    ):
        """
        Initializes graph layer with GCNBlock, a base sample grid, and edges.

        Args:
            settings (ModelSettings): Configuration settings for the global graph layer.
            points_per_dim: Number of points per dimension, in sampling grid
            out_dim_graph: Output dimension of graph layers (6 or 7 see 'get_affine_matrix')
            shape_template: Initial anatomical shape for the model
        """
        super().__init__()

        self.gcn_block = GCNBlock(
            settings=settings, mean_aggregate=True, out_dim=out_dim_graph
        )

        # Register sample grid (not learnt)
        sample_grid, n_points = create_uniform_3d_grid(n_points=points_per_dim)
        self.register_buffer("sample_grid", sample_grid)
        # Initialize edges for 'sample_grid' (learnable):
        self.edges = get_initial_edges(n_points=n_points)

        # Register initial shape (not learnt)
        self.register_buffer("shape_template", shape_template)

        self.detached_feature_map = detached_feature_map

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass for graph layer.
        Forward pass computes features for the graph nodes from feature maps.

        Args:
            x: Input tensor of shape (B, feat_maps, X, Y, Z)
                Encoding feature maps, e.g. from 3D CNN.
        Returns:
            Transformed 'shape_template' per sample in batch.
            Dictionary with affine transform parameters applied to 'shape_template'
        """

        batch_size = x["cnn_prediction"].shape[0]

        shape_template_batch = self.shape_template.expand(batch_size, -1, -1)
        sample_grid_batch = self.sample_grid.expand(batch_size, -1, -1, -1, -1)
        edges_batch = repeat(self.edges, "1 ... -> b ...", b=batch_size)

        # CNN feature channels with spatial loss (detached, if specified):
        cnn_prediction = (
            x["cnn_prediction"].detach()
            if self.detached_feature_map
            else x["cnn_prediction"]
        )
        # Remaining feature channels:
        feature_list = [
            value for key, value in x.items() if key.endswith(("_features", "_feature"))
        ]
        # All features from CNN:
        feature_map = torch.cat([cnn_prediction] + feature_list, dim=1)

        # Get graph layer output:
        features = sample_coords_from_encoder_features(
            feature_map=feature_map, coordinates=sample_grid_batch
        )
        graph_output = self.gcn_block(features=features, edges=edges_batch)
        assert graph_output.ndim == 2
        # Apply transforms to 'initial_shape':
        affine_matrix, transform_params = get_affine_matrix(graph_output)
        shape_batch = apply_transform(
            affine_matrix=affine_matrix, points=shape_template_batch
        )

        result = {
            **x,
            "gcn_prediction_global": shape_batch,
            "transform_params": transform_params,
            "graph_output": graph_output,
        }  # type: ignore

        return result


class LocalGraphLayer(nn.Module):
    def __init__(
        self,
        settings: GraphModelSettings,
        num_local_layers: int,
        shape_template: torch.Tensor,
        detached_feature_map: bool = False,
    ):
        """
        Initializes graph layer with GCNBlock, a base shape/graph, and edges.

        Args:
            settings (ModelSettings): Configuration settings for the graph layer.
            num_local_layers: Number of local layers to stack
            shape_template: Shape template (for num elements in adjacency matrix)
        """
        super().__init__()
        # Get 'num_local_layers' GCN Blocks:
        self.graph_layers = self._get_local_gcn_layers(
            settings=settings, num_layers=num_local_layers, out_dim=3
        )
        n_points_shape_template = shape_template.shape[0]
        # Initialize edges for the graph (learnable):
        self.edges = get_initial_edges(n_points=n_points_shape_template)
        self.detached_feature_map = detached_feature_map

    @staticmethod
    def _get_local_gcn_layers(
        settings: GraphModelSettings,
        num_layers: int,
        out_dim=3,
    ) -> nn.ModuleList:
        """Returns 'num_layers' 'GCNBlock' instances.

        Args:
            settings: Configuration for graph layer
            num_layers: Number of local layers to stack
            out_dim: Dimensionality of the output
        """
        if num_layers <= 0:
            raise AttributeError("Number of graph layers to be greater than zero.")

        layer_sharing = False
        if layer_sharing:
            layer = GCNBlock(
                settings=settings,
                mean_aggregate=False,
                out_dim=out_dim,
            )
            return nn.ModuleList([layer for _ in range(num_layers)])

        else:
            return nn.ModuleList(
                [
                    GCNBlock(
                        settings=settings,
                        mean_aggregate=False,
                        out_dim=out_dim,
                    )
                    for _ in range(num_layers)
                ]
            )

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass for graph layer.
        Forward pass computes features for the graph nodes from feature maps.

        Args:
            x: Input tensor of shape (B, feat_maps, X, Y, Z)
            initial_shape: Coarse prediction of the shape
        """
        batch_size = x["cnn_prediction"].shape[0]
        current_coordinates = x["gcn_prediction_global"]

        # CNN feature channels with spatial loss (detached, if specified):
        cnn_prediction = (
            x["cnn_prediction"].detach()
            if self.detached_feature_map
            else x["cnn_prediction"]
        )
        # Remaining feature channels:
        feature_list = [
            value for key, value in x.items() if key.endswith(("_features", "_feature"))
        ]
        # All features from CNN:
        feature_map = torch.cat([cnn_prediction] + feature_list, dim=1)

        layer_output = {}

        for layer_i, layer in enumerate(self.graph_layers):
            features = sample_coords_from_encoder_features(
                feature_map=feature_map, coordinates=current_coordinates
            )
            edges_batch = repeat(self.edges, "1 ... -> b ...", b=batch_size)
            graph_output = layer(features, edges=edges_batch)

            assert graph_output.ndim == 3
            layer_output[f"layer_{layer_i}"] = graph_output

            current_coordinates = current_coordinates + graph_output

        return {  # this has invalid mypy type hint
            **x,
            **layer_output,
            "gcn_prediction": current_coordinates,
        }  # type: ignore
