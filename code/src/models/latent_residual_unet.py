from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


def _group_count(channels: int, preferred: int = 8) -> int:
    for groups in (preferred, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class Conv3dResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.skip(x) + self.net(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )
        self.block = Conv3dResBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.block = Conv3dResBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        x = self.proj(x)
        return self.block(torch.cat([x, skip], dim=1))


@dataclass
class LatentResidualOutput:
    delta: torch.Tensor
    soft_valid: torch.Tensor


class WarpMaskLatentResidualUNet(nn.Module):
    """
    Predict z_gt - z_warp from the warped latent and a soft valid mask.

    Inputs:
      warp_latents: [B, C, F, H, W]
      soft_valid:   [B, 1, F, H, W], continuous valid score in [0, 1]

    Output:
      delta: [B, C, F, H, W]
    """

    def __init__(
        self,
        latent_channels: int = 16,
        base_channels: int = 96,
        channel_mults: tuple[int, int, int] = (1, 2, 4),
        zero_init_output: bool = True,
    ):
        super().__init__()
        if latent_channels <= 0:
            raise ValueError(f"latent_channels must be positive, got {latent_channels}")
        if base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {base_channels}")
        if len(channel_mults) != 3:
            raise ValueError(f"channel_mults must have length 3, got {channel_mults}")

        self.latent_channels = int(latent_channels)
        self.base_channels = int(base_channels)
        self.channel_mults = tuple(int(mult) for mult in channel_mults)
        c0 = self.base_channels * self.channel_mults[0]
        c1 = self.base_channels * self.channel_mults[1]
        c2 = self.base_channels * self.channel_mults[2]

        self.input_proj = nn.Sequential(
            nn.Conv3d(self.latent_channels + 1, c0, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(c0), c0),
            nn.SiLU(),
        )
        self.enc0 = Conv3dResBlock(c0, c0)
        self.down1 = DownBlock(c0, c1)
        self.down2 = DownBlock(c1, c2)
        self.mid = nn.Sequential(
            Conv3dResBlock(c2, c2),
            Conv3dResBlock(c2, c2),
        )
        self.up1 = UpBlock(c2, c1, c1)
        self.up0 = UpBlock(c1, c0, c0)
        self.output_proj = nn.Sequential(
            nn.GroupNorm(_group_count(c0), c0),
            nn.SiLU(),
            nn.Conv3d(c0, self.latent_channels, kernel_size=3, padding=1),
        )

        if zero_init_output:
            last = self.output_proj[-1]
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    @staticmethod
    def prepare_soft_valid(soft_valid: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if soft_valid.dim() != 5:
            raise ValueError(f"soft_valid must be [B,C,F,H,W], got {tuple(soft_valid.shape)}")
        if target.dim() != 5:
            raise ValueError(f"target must be [B,C,F,H,W], got {tuple(target.shape)}")
        if soft_valid.size(0) != target.size(0):
            raise ValueError(
                f"batch mismatch: soft_valid B={soft_valid.size(0)} vs target B={target.size(0)}"
            )

        if soft_valid.size(1) != 1:
            soft_valid = soft_valid.mean(dim=1, keepdim=True)
        soft_valid = soft_valid.to(device=target.device, dtype=target.dtype)
        if soft_valid.shape[-3:] != target.shape[-3:]:
            soft_valid = F.interpolate(
                soft_valid.float(),
                size=target.shape[-3:],
                mode="trilinear",
                align_corners=False,
            ).to(device=target.device, dtype=target.dtype)
        return soft_valid.clamp(0.0, 1.0)

    def forward(
        self,
        warp_latents: torch.Tensor,
        soft_valid: torch.Tensor,
        return_dict: bool = False,
    ):
        if warp_latents.dim() != 5:
            raise ValueError(f"warp_latents must be [B,C,F,H,W], got {tuple(warp_latents.shape)}")
        if warp_latents.size(1) != self.latent_channels:
            raise ValueError(
                f"warp_latents has {warp_latents.size(1)} channels, expected {self.latent_channels}"
            )

        soft_valid = self.prepare_soft_valid(soft_valid, warp_latents)
        x = torch.cat([warp_latents, soft_valid], dim=1)
        h0 = self.enc0(self.input_proj(x))
        h1 = self.down1(h0)
        h2 = self.down2(h1)
        hidden = self.mid(h2)
        hidden = self.up1(hidden, h1)
        hidden = self.up0(hidden, h0)
        delta = self.output_proj(hidden)

        if return_dict:
            return LatentResidualOutput(delta=delta, soft_valid=soft_valid)
        return delta

    def config_dict(self) -> Dict[str, object]:
        return {
            "latent_channels": self.latent_channels,
            "base_channels": self.base_channels,
            "channel_mults": list(self.channel_mults),
        }
