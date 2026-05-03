"""MeshGraphNet for 3D field prediction on voxel graphs.

The structured grid is subsampled to reduce the number of nodes,
then predictions are upsampled back to full resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel, register_model

try:
    from torch_geometric.nn import MessagePassing
    from torch_geometric.data import Data, Batch

    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    MessagePassing = nn.Module  # fallback for type checking


def _build_grid_edges(D: int, H: int, W: int) -> torch.Tensor:
    """Build 6-connected edge index for a 3D grid.

    Returns (2, n_edges) long tensor.
    """
    n = D * H * W
    edges = []

    def idx(d, h, w):
        return d * H * W + h * W + w

    for d in range(D):
        for h in range(H):
            for w in range(W):
                i = idx(d, h, w)
                if d + 1 < D:
                    edges.append((i, idx(d + 1, h, w)))
                    edges.append((idx(d + 1, h, w), i))
                if h + 1 < H:
                    edges.append((i, idx(d, h + 1, w)))
                    edges.append((idx(d, h + 1, w), i))
                if w + 1 < W:
                    edges.append((i, idx(d, h, w + 1)))
                    edges.append((idx(d, h, w + 1), i))

    return torch.tensor(edges, dtype=torch.long).T


class EdgeConv3D(MessagePassing if HAS_PYG else nn.Module):
    """Message-passing layer with edge features."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        if HAS_PYG:
            super().__init__(aggr="mean")
        else:
            super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if HAS_PYG:
            out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        else:
            # Fallback without torch_geometric
            src, dst = edge_index
            msg = self.edge_mlp(torch.cat([x[src], x[dst], edge_attr], dim=-1))
            out = torch.zeros_like(x)
            out.index_add_(0, dst, msg)
            count = torch.zeros(x.size(0), 1, device=x.device)
            count.index_add_(0, dst, torch.ones(src.size(0), 1, device=x.device))
            out = out / count.clamp(min=1)

        out = self.node_mlp(torch.cat([x, out], dim=-1))
        return self.norm(x + out)  # residual

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


@register_model
class MeshGraphNet3D(BaseModel):
    """MeshGraphNet operating on a subsampled voxel grid.

    Pipeline:
    1. Subsample input fields (e.g., 2× per dim → 64×64×16)
    2. Build 6-connected graph
    3. Encode-process-decode with message passing
    4. Trilinear upsample back to original resolution
    """

    name = "meshgraphnet"

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 5,
        scalar_dim: int = 5,
        hidden_dim: int = 128,
        n_layers: int = 6,
        subsample_factor: int = 4,
        grid_shape: tuple[int, ...] = (128, 128, 32),
        **kwargs,
    ):
        super().__init__()
        self.subsample_factor = subsample_factor
        self.grid_shape = grid_shape
        self.out_channels = out_channels
        sf = subsample_factor
        self.sub_shape = (grid_shape[0] // sf, grid_shape[1] // sf, grid_shape[2] // sf)

        # Node feature encoder: input channels + 3 (coordinates) + scalar_dim
        node_in = in_channels + 3 + scalar_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Edge feature encoder: relative position (3D)
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Message passing layers
        self.mp_layers = nn.ModuleList([
            EdgeConv3D(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels),
        )

        # Precompute edges and relative positions for the subsampled grid
        D, H, W = self.sub_shape
        self._edge_index = _build_grid_edges(D, H, W)

        # Precompute normalized coords and relative positions
        zz, yy, xx = torch.meshgrid(
            torch.linspace(0, 1, D), torch.linspace(0, 1, H), torch.linspace(0, 1, W),
            indexing="ij",
        )
        self._coords = torch.stack([zz, yy, xx], dim=-1).reshape(-1, 3)

        src, dst = self._edge_index
        self._rel_pos = self._coords[dst] - self._coords[src]

    def forward(self, input_dict: dict) -> dict:
        fields = input_dict["fields"]  # (B, C_in, D, H, W)
        scalars = input_dict.get("scalar_params")  # (B, S)

        B = fields.shape[0]
        sf = self.subsample_factor
        D, H, W = self.sub_shape

        # Subsample via average pooling
        sub_fields = F.avg_pool3d(fields, kernel_size=sf, stride=sf)  # (B, C, D', H', W')

        # Flatten to node features: (B, N, C_in)
        node_feats = sub_fields.flatten(2).transpose(1, 2)  # (B, N, C_in)
        N = node_feats.shape[1]

        # Add coordinates
        coords = self._coords.to(fields.device).unsqueeze(0).expand(B, -1, -1)
        node_feats = torch.cat([node_feats, coords], dim=-1)  # (B, N, C_in + 3)

        # Add scalar params (broadcast to all nodes)
        if scalars is not None:
            s = scalars.unsqueeze(1).expand(-1, N, -1)  # (B, N, S)
            node_feats = torch.cat([node_feats, s], dim=-1)  # (B, N, C_in + 3 + S)

        # Move edges to device
        edge_index = self._edge_index.to(fields.device)
        rel_pos = self._rel_pos.to(fields.device)

        # Process each sample (graph batching)
        all_outputs = []
        for b in range(B):
            x = self.node_encoder(node_feats[b])  # (N, H)
            edge_feat = self.edge_encoder(rel_pos)  # (E, H)

            for mp in self.mp_layers:
                x = mp(x, edge_index, edge_feat)

            out = self.decoder(x)  # (N, C_out)
            all_outputs.append(out)

        output = torch.stack(all_outputs)  # (B, N, C_out)

        # Reshape to subsampled grid
        output = output.transpose(1, 2).view(B, self.out_channels, D, H, W)

        # Upsample to original resolution
        output = F.interpolate(output, size=self.grid_shape, mode="trilinear", align_corners=False)

        return {"fields": output}
