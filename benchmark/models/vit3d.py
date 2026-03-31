"""3D Vision Transformer for volumetric field prediction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import BaseModel, register_model


class PatchEmbed3D(nn.Module):
    """Split a 3D volume into non-overlapping patches and project."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: tuple[int, ...] = (8, 8, 8)):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, D, H, W) → (B, N_patches, embed_dim)."""
        x = self.proj(x)  # (B, E, D', H', W')
        return x.flatten(2).transpose(1, 2)  # (B, N, E)


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm."""

    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(*[self.norm1(x)] * 3)[0]
        x = x + self.mlp(self.norm2(x))
        return x


@register_model
class ViT3D(BaseModel):
    """3D Vision Transformer: patchify → transformer → unpatchify.

    Scalar params are injected as a learnable condition token prepended to the
    patch sequence.
    """

    name = "vit3d"

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 5,
        scalar_dim: int = 5,
        patch_size: tuple[int, ...] = (8, 8, 8),
        embed_dim: int = 256,
        depth: int = 8,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        grid_shape: tuple[int, ...] = (128, 128, 32),
        **kwargs,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.grid_shape = grid_shape
        self.out_channels = out_channels
        self.embed_dim = embed_dim

        # Number of patches per dimension
        self.grid_patches = tuple(g // p for g, p in zip(grid_shape, patch_size))
        n_patches = math.prod(self.grid_patches)

        # Patch embedding
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, patch_size)

        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, embed_dim) * 0.02)

        # Scalar conditioning token
        self.scalar_proj = nn.Linear(scalar_dim, embed_dim)
        self.cond_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Decoder: project each patch token back to patch volume
        patch_volume = math.prod(patch_size) * out_channels
        self.head = nn.Linear(embed_dim, patch_volume)

    def forward(self, input_dict: dict) -> dict:
        x = input_dict["fields"]  # (B, C_in, D, H, W)
        scalars = input_dict.get("scalar_params")

        B = x.shape[0]

        # Patchify
        tokens = self.patch_embed(x) + self.pos_embed  # (B, N, E)

        # Prepend condition token
        if scalars is not None:
            cond = self.cond_token.expand(B, -1, -1) + self.scalar_proj(scalars).unsqueeze(1)
            tokens = torch.cat([cond, tokens], dim=1)  # (B, 1+N, E)

        # Transformer
        tokens = self.norm(self.blocks(tokens))

        # Remove condition token
        if scalars is not None:
            tokens = tokens[:, 1:]  # (B, N, E)

        # Unpatchify: project each token to a patch volume
        patches = self.head(tokens)  # (B, N, C_out * pD * pH * pW)
        pD, pH, pW = self.patch_size
        gD, gH, gW = self.grid_patches

        # Reshape: (B, gD, gH, gW, C_out, pD, pH, pW) → (B, C_out, D, H, W)
        patches = patches.view(B, gD, gH, gW, self.out_channels, pD, pH, pW)
        output = patches.permute(0, 4, 1, 5, 2, 6, 3, 7)  # (B, C, gD, pD, gH, pH, gW, pW)
        output = output.reshape(B, self.out_channels, *self.grid_shape)

        return {"fields": output}
