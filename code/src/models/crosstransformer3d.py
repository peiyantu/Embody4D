
from typing import Any, Dict, Optional, Tuple, Union

import math
import os
import json
import torch
import glob
import torch.nn.functional as F
from torch import nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import is_torch_version, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import (
    AttentionProcessor,
    CogVideoXAttnProcessor2_0,
    FusedCogVideoXAttnProcessor2_0,
)
from diffusers.models.embeddings import (
    TimestepEmbedding,
    Timesteps,
    get_3d_sincos_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _get_3d_sincos_pos_embed_tensor(*args, **kwargs):
    try:
        pos_embedding = get_3d_sincos_pos_embed(*args, output_type="pt", **kwargs)
    except TypeError:
        pos_embedding = get_3d_sincos_pos_embed(*args, **kwargs)
    if not isinstance(pos_embedding, torch.Tensor):
        pos_embedding = torch.from_numpy(pos_embedding)
    return pos_embedding


class FFNResidualExpertRouter(nn.Module):
    def __init__(
        self,
        dim: int,
        rank: int,
        num_experts: int = 3,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")

        self.num_experts = int(num_experts)
        self.scale = float(scale)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, rank),
                    nn.SiLU(),
                    nn.Linear(rank, dim),
                )
                for _ in range(self.num_experts)
            ]
        )
        for expert in self.experts:
            last = expert[-1]
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        route_weights: torch.Tensor,
        scale: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if route_weights.dim() != 3 or route_weights.size(-1) != self.num_experts:
            raise ValueError(
                "route_weights must be [B, N, num_experts], got "
                f"{tuple(route_weights.shape)}"
            )
        if route_weights.shape[:2] != hidden_states.shape[:2]:
            raise ValueError(
                "route_weights token shape must match hidden_states, got "
                f"{tuple(route_weights.shape[:2])} vs {tuple(hidden_states.shape[:2])}"
            )

        route_weights = route_weights.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        route_weights = route_weights.clamp_min(0.0)
        route_weights = route_weights / route_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)

        expert_delta = hidden_states.new_zeros(hidden_states.shape)
        for expert_idx, expert in enumerate(self.experts):
            expert_delta = expert_delta + expert(hidden_states) * route_weights[
                :, :, expert_idx : expert_idx + 1
            ]

        if scale is None:
            active_scale = self.scale
        elif torch.is_tensor(scale):
            active_scale = scale.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            while active_scale.dim() < expert_delta.dim():
                active_scale = active_scale.unsqueeze(-1)
        else:
            active_scale = float(scale)
        return expert_delta * active_scale


class CogVideoXPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        embed_dim: int = 1920,
        text_embed_dim: int = 4096,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=patch_size,
            bias=bias,
        )
        self.text_proj = nn.Linear(text_embed_dim, embed_dim)

    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor):
        r"""
        Args:
            text_embeds (`torch.Tensor`):
                Input text embeddings. Expected shape: (batch_size, seq_length, embedding_dim).
            image_embeds (`torch.Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
        """
        text_embeds = self.text_proj(text_embeds)

        batch, num_frames, channels, height, width = image_embeds.shape
        image_embeds = image_embeds.reshape(-1, channels, height, width)
        # 1. 先把数据搬运到 GPU (必须加在 self.proj 之前！)
        if image_embeds.device != self.proj.weight.device:
            image_embeds = image_embeds.to(self.proj.weight.device)

        # 2. 确保数据类型也对齐 (防止一个是 float32 一个是 bfloat16)
        if image_embeds.dtype != self.proj.weight.dtype:
            image_embeds = image_embeds.to(dtype=self.proj.weight.dtype)

        # 3. 然后再执行原来的代码
        image_embeds = self.proj(image_embeds)
        image_embeds = image_embeds.view(batch, num_frames, *image_embeds.shape[1:])
        image_embeds = image_embeds.flatten(3).transpose(
            2, 3
        )  # [batch, num_frames, height x width, channels]
        image_embeds = image_embeds.flatten(
            1, 2
        )  # [batch, num_frames x height x width, channels]

        embeds = torch.cat(
            [text_embeds, image_embeds], dim=1
        ).contiguous()  # [batch, seq_length + num_frames x height x width, channels]
        return embeds


class RefPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        embed_dim: int = 1920,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=patch_size,
            bias=bias,
        )

    def forward(self, image_embeds: torch.Tensor):
        r"""
        Args:
            image_embeds (`torch.Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
        """
        batch, num_frames, channels, height, width = image_embeds.shape
        image_embeds = image_embeds.reshape(-1, channels, height, width)
        image_embeds = self.proj(image_embeds)
        image_embeds = image_embeds.view(batch, num_frames, *image_embeds.shape[1:])
        image_embeds = image_embeds.flatten(3).transpose(
            2, 3
        )  # [batch, num_frames, height x width, channels]
        image_embeds = image_embeds.flatten(
            1, 2
        )  # [batch, num_frames x height x width, channels]
        return image_embeds


class FrontConditionAdapter(nn.Module):
    def __init__(
        self,
        *,
        patch_size: int = 2,
        embed_dim: int = 1920,
        use_boundary: bool = True,
        strength_init: float = 1.0,
        gate_init: float = 0.05,
        gate_max: float = 0.25,
        video_scale_init: float = 0.0,
        front_scale_init: float = 0.0,
        boundary_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.use_boundary = bool(use_boundary)
        self.gate_init = float(gate_init)
        self.gate_max = float(gate_max)
        in_channels = 2 if self.use_boundary else 1
        self.delta_proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=patch_size,
            bias=False,
        )
        self.strength = nn.Parameter(torch.tensor(float(strength_init), dtype=torch.float32))
        self.gate_bias = nn.Parameter(
            torch.tensor(self._gate_bias_from_init(gate_init, gate_max), dtype=torch.float32)
        )
        self.video_scale = nn.Parameter(torch.tensor(float(video_scale_init), dtype=torch.float32))
        self.front_scale = nn.Parameter(torch.tensor(float(front_scale_init), dtype=torch.float32))
        self.boundary_scale = nn.Parameter(torch.tensor(float(boundary_scale_init), dtype=torch.float32))
        self._last_log_values: Dict[str, torch.Tensor] = {}
        nn.init.zeros_(self.delta_proj.weight)

    @staticmethod
    def _gate_bias_from_init(gate_init: float, gate_max: float) -> float:
        if gate_max <= 0:
            return -20.0
        prob = float(gate_init) / float(gate_max)
        prob = min(max(prob, 1e-4), 1.0 - 1e-4)
        return math.log(prob / (1.0 - prob))

    @staticmethod
    def _align_tokens(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
        token_length = tensor.shape[1]
        if token_length < target_length:
            tensor = F.pad(tensor, (0, 0, 0, target_length - token_length))
        elif token_length > target_length:
            tensor = tensor[:, :target_length]
        return tensor

    def forward(
        self,
        front_latents: torch.Tensor,
        video_tokens: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ):
        if front_latents.dim() != 5:
            raise ValueError(
                f"front_latents must be 5D, got shape {tuple(front_latents.shape)}"
            )
        if video_tokens.dim() != 3:
            raise ValueError(
                f"video_tokens must be 3D, got shape {tuple(video_tokens.shape)}"
            )
        if front_latents.shape[1] in (1, 3):
            front = front_latents
        elif front_latents.shape[2] in (1, 3):
            front = front_latents.permute(0, 2, 1, 3, 4)
        else:
            raise ValueError(
                "Unable to infer channel dimension for front_latents. "
                f"Expected shape [B,C,F,H,W] or [B,F,C,H,W], got {tuple(front_latents.shape)}"
            )

        front = front.to(device=device, dtype=dtype)
        if front.shape[1] > 1:
            front = front.mean(dim=1, keepdim=True)
        elif front.shape[1] != 1:
            front = front[:, :1]
        front = front.clamp(0.0, 1.0)

        boundary = torch.zeros_like(front)
        if self.use_boundary:
            dilated = F.max_pool3d(
                front, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
            )
            eroded = -F.max_pool3d(
                -front, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
            )
            boundary = (dilated - eroded).clamp(0.0, 1.0)
            front_features = torch.cat([front, boundary], dim=1)
        else:
            front_features = front

        batch, channels, num_frames, height, width = front_features.shape
        front_features = front_features.permute(0, 2, 1, 3, 4).reshape(
            batch * num_frames, channels, height, width
        )
        if front_features.dtype != self.delta_proj.weight.dtype:
            front_features = front_features.to(dtype=self.delta_proj.weight.dtype)
        cond_delta = self.delta_proj(front_features)

        front_for_pool = front.permute(0, 2, 1, 3, 4).reshape(
            batch * num_frames, 1, height, width
        )
        boundary_for_pool = boundary.permute(0, 2, 1, 3, 4).reshape(
            batch * num_frames, 1, height, width
        )
        if front_for_pool.dtype != cond_delta.dtype:
            front_for_pool = front_for_pool.to(dtype=cond_delta.dtype)
            boundary_for_pool = boundary_for_pool.to(dtype=cond_delta.dtype)

        front_occ = F.avg_pool2d(
            front_for_pool, kernel_size=self.patch_size, stride=self.patch_size
        )
        boundary_occ = F.avg_pool2d(
            boundary_for_pool, kernel_size=self.patch_size, stride=self.patch_size
        )

        cond_delta = cond_delta.view(batch, num_frames, *cond_delta.shape[1:])
        cond_delta = cond_delta.flatten(3).transpose(2, 3).flatten(1, 2)
        front_occ = front_occ.view(batch, num_frames, *front_occ.shape[1:])
        front_occ = front_occ.flatten(3).transpose(2, 3).flatten(1, 2)
        boundary_occ = boundary_occ.view(batch, num_frames, *boundary_occ.shape[1:])
        boundary_occ = boundary_occ.flatten(3).transpose(2, 3).flatten(1, 2)

        target_length = video_tokens.shape[1]
        cond_delta = self._align_tokens(cond_delta, target_length)
        front_occ = self._align_tokens(front_occ, target_length)
        boundary_occ = self._align_tokens(boundary_occ, target_length)

        video_score = video_tokens.detach().float().abs().mean(dim=-1, keepdim=True)
        video_mean = video_score.mean(dim=1, keepdim=True)
        video_std = video_score.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        video_score = (video_score - video_mean) / video_std

        gate_logits = (
            self.gate_bias.float()
            + self.video_scale.float() * video_score
            + self.front_scale.float() * front_occ.float()
            + self.boundary_scale.float() * boundary_occ.float()
        )
        gate = max(0.0, self.gate_max) * torch.sigmoid(gate_logits)
        strength = self.strength.to(device=cond_delta.device, dtype=cond_delta.dtype)
        applied = cond_delta * gate.to(dtype=cond_delta.dtype) * strength

        video_norm = video_tokens.detach().float().norm().clamp_min(1e-12)
        applied_norm = applied.detach().float().norm()
        self._last_log_values = {
            "gate_mean": gate.detach().float().mean(),
            "gate_max": gate.detach().float().max(),
            "applied_norm": applied_norm,
            "applied_to_video_norm_ratio": applied_norm / video_norm,
        }
        return applied


@maybe_allow_in_graph
class CogVideoXBlock(nn.Module):
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        ffn_expert_rank: Optional[int] = None,
        ffn_expert_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = int(dim)

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(
            time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True
        )

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(
            time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True
        )

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        self.ffn_expert_router = None
        if ffn_expert_rank is not None:
            self.enable_ffn_expert_router(ffn_expert_rank, ffn_expert_scale)

    def enable_ffn_expert_router(self, rank: int, scale: float = 1.0) -> None:
        if self.ffn_expert_router is None:
            self.ffn_expert_router = FFNResidualExpertRouter(
                dim=self.dim,
                rank=int(rank),
                scale=float(scale),
            )
            ref_param = next(self.parameters())
            self.ffn_expert_router.to(device=ref_param.device, dtype=ref_param.dtype)
        else:
            self.ffn_expert_router.scale = float(scale)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ffn_expert_routes: Optional[torch.Tensor] = None,
        ffn_expert_scale: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = (
            self.norm1(hidden_states, encoder_hidden_states, temb)
        )

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = (
            encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states
        )

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = (
            self.norm2(hidden_states, encoder_hidden_states, temb)
        )

        # feed-forward
        norm_hidden_states = torch.cat(
            [norm_encoder_hidden_states, norm_hidden_states], dim=1
        )
        ff_output = self.ff(norm_hidden_states)

        ff_hidden_states = ff_output[:, text_seq_length:]
        if ffn_expert_routes is not None:
            if self.ffn_expert_router is None:
                raise RuntimeError(
                    "ffn_expert_routes were provided, but this block has no "
                    "ffn_expert_router. Call enable_ffn_expert_routers first."
                )
            ff_hidden_states = ff_hidden_states + self.ffn_expert_router(
                ff_hidden_states,
                ffn_expert_routes,
                scale=ffn_expert_scale,
            )

        hidden_states = hidden_states + gate_ff * ff_hidden_states
        encoder_hidden_states = (
            encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]
        )

        return hidden_states, encoder_hidden_states


def reshape_tensor(x, heads):
    """
    Reshapes the input tensor for multi-head attention.

    Args:
        x (torch.Tensor): The input tensor with shape (batch_size, length, width).
        heads (int): The number of attention heads.

    Returns:
        torch.Tensor: The reshaped tensor, with shape (batch_size, heads, length, width).
    """
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1)
    x = x.transpose(1, 2)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverCrossAttention(nn.Module):
    """

    Args:
        dim (int): Dimension of the input latent and output. Default is 3072.
        dim_head (int): Dimension of each attention head. Default is 128.
        heads (int): Number of attention heads. Default is 16.
        kv_dim (int): Dimension of the key/value input, allowing flexible cross-attention. Default is 2048.

    Attributes:
        scale (float): Scaling factor used in dot-product attention for numerical stability.
        norm1 (nn.LayerNorm): Layer normalization applied to the input image features.
        norm2 (nn.LayerNorm): Layer normalization applied to the latent features.
        to_q (nn.Linear): Linear layer for projecting the latent features into queries.
        to_kv (nn.Linear): Linear layer for projecting the input features into keys and values.
        to_out (nn.Linear): Linear layer for outputting the final result after attention.

    """

    def __init__(
        self,
        *,
        dim=3072,
        dim_head=128,
        heads=16,
        kv_dim=2048,
        enable_front_guidance: bool = False,
        target_guidance_scale: float = 0.6,
        boost_strength: float = 5.0,
        front_guidance_mode: str = "blend",
        front_guidance_min_area: float = 0.0,
        front_guidance_max_area: float = 1.0,
        front_guidance_dropout_prob: float = 0.0,
        learnable_front_guidance: bool = False,
        front_guidance_alpha_max: float = 0.15,
        front_guidance_alpha_init: float = 1e-3,
        adaptive_front_guidance: bool = False,
        front_guidance_gate_max: float = 0.03,
        front_guidance_gate_init: float = 0.0,
        front_guidance_gate_floor: float = 0.0,
        front_memory_high_threshold: float = 0.55,
        front_memory_quality_threshold: float = 0.01,
        front_memory_noise_threshold: float = 0.08,
        front_memory_min_alignment: float = 0.0,
        front_memory_quality_power: float = 1.0,
        front_memory_residual_ratio_max: float = -1.0,
        front_memory_query_mask_power: float = 0.0,
        front_memory_query_mask_floor: float = 0.0,
        front_memory_train_gate_gradient_multiplier: float = 1.0,
        front_memory_max_area_std: float = -1.0,
        front_memory_min_area_std: float = -1.0,
        front_memory_min_valid_frame_ratio: float = 0.0,
        front_memory_max_area_jump: float = -1.0,
        front_memory_max_spatial_noise: float = -1.0,
        front_memory_min_high_active_ratio: float = 0.0,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads
        self.inner_dim = inner_dim
        self.enable_front_guidance = bool(enable_front_guidance)
        self.target_guidance_scale = float(target_guidance_scale)
        self.boost_strength = float(boost_strength)
        self.front_guidance_mode = str(front_guidance_mode)
        self.front_guidance_min_area = float(front_guidance_min_area)
        self.front_guidance_max_area = float(front_guidance_max_area)
        self.front_guidance_dropout_prob = float(front_guidance_dropout_prob)
        self.learnable_front_guidance = bool(learnable_front_guidance)
        self.front_guidance_alpha_max = float(front_guidance_alpha_max)
        self.front_guidance_alpha_init = float(front_guidance_alpha_init)
        self.adaptive_front_guidance = bool(adaptive_front_guidance)
        self.front_guidance_gate_max = float(front_guidance_gate_max)
        self.front_guidance_gate_init = float(front_guidance_gate_init)
        self.front_guidance_gate_floor = float(front_guidance_gate_floor)
        self.front_memory_high_threshold = float(front_memory_high_threshold)
        self.front_memory_quality_threshold = float(front_memory_quality_threshold)
        self.front_memory_noise_threshold = float(front_memory_noise_threshold)
        self.front_memory_min_alignment = float(front_memory_min_alignment)
        self.front_memory_quality_power = float(front_memory_quality_power)
        self.front_memory_residual_ratio_max = float(front_memory_residual_ratio_max)
        self.front_memory_query_mask_power = float(front_memory_query_mask_power)
        self.front_memory_query_mask_floor = float(front_memory_query_mask_floor)
        self.front_memory_train_gate_gradient_multiplier = float(
            front_memory_train_gate_gradient_multiplier
        )
        self.front_memory_max_area_std = float(front_memory_max_area_std)
        self.front_memory_min_area_std = float(front_memory_min_area_std)
        self.front_memory_min_valid_frame_ratio = float(
            front_memory_min_valid_frame_ratio
        )
        self.front_memory_max_area_jump = float(front_memory_max_area_jump)
        self.front_memory_max_spatial_noise = float(front_memory_max_spatial_noise)
        self.front_memory_min_high_active_ratio = float(
            front_memory_min_high_active_ratio
        )
        self._last_front_guidance_gate = None
        self._last_front_guidance_gate_for_loss = None
        self._last_front_guidance_area = None
        self._last_front_guidance_valid_area = None
        self._last_front_memory_quality = None
        self._last_front_memory_noise = None
        self._last_front_memory_sample_gate = None
        self._last_front_memory_area_std = None
        self._last_front_memory_max_area_jump = None
        self._last_front_memory_valid_frame_ratio = None
        self._last_front_memory_high_active_ratio = None
        self._last_front_memory_alignment = None
        self._last_front_memory_core_area = None
        self._last_front_memory_boundary_area = None
        self._last_front_memory_enabled = None
        self._last_front_memory_residual_ratio = None
        self._last_front_memory_query_gate = None
        self._last_front_memory_train_gate_gradient_multiplier = None
        if self.learnable_front_guidance:
            self.init_front_guidance_alpha()
        if self.adaptive_front_guidance:
            self.init_adaptive_front_guidance()

        # Layer normalization to stabilize training
        self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
        self.norm2 = nn.LayerNorm(dim)

        # Linear transformations to produce queries, keys, and values
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(
            dim if kv_dim is None else kv_dim, inner_dim * 2, bias=False
        )
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        if self.front_guidance_mode == "foreground_memory":
            self.init_front_memory()

    def forward(
        self,
        x,
        latents,
        ref_mask=None,
        train_progress: float = 1.0,
        ref_token_shape=None,
    ):
        """

        Args:
            x (torch.Tensor): Input image features with shape (batch_size, n1, D), where:
                - batch_size (b): Number of samples in the batch.
                - n1: Sequence length (e.g., number of patches or tokens).
                - D: Feature dimension.

            latents (torch.Tensor): Latent feature representations with shape (batch_size, n2, D), where:
                - n2: Number of latent elements.

        Returns:
            torch.Tensor: Attention-modulated features with shape (batch_size, n2, D).

        """
        # Apply layer normalization to the input image and latent features
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, seq_len, _ = latents.shape

        # Compute queries, keys, and values
        q = self.to_q(latents)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        # Reshape tensors to split into attention heads
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # Avoid materializing a [B, heads, query_tokens, key_tokens] attention
        # matrix. At 480x832/49f this matrix is tens of GB.
        out_global = F.scaled_dot_product_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            dropout_p=0.0,
            is_causal=False,
        )
        out_extra = None

        # Keep strict no-op behavior unless guidance is explicitly enabled and ref mask is provided.
        if self.enable_front_guidance and ref_mask is not None:
            if ref_mask.dim() > 2:
                ref_mask = ref_mask.flatten(1)

            key_len = k.shape[2]
            if ref_mask.shape[1] < key_len:
                ref_mask = F.pad(ref_mask, (0, key_len - ref_mask.shape[1]))
            elif ref_mask.shape[1] > key_len:
                ref_mask = ref_mask[:, :key_len]
            ref_mask = ref_mask.to(device=q.device, dtype=torch.float32)

            adaptive_gate = None
            if self.front_guidance_mode == "foreground_memory":
                alpha = None
                should_compute_guided = (
                    self.front_guidance_gate_max > 0 and self.boost_strength != 0
                )
            elif self.adaptive_front_guidance:
                alpha = None
                should_compute_guided = self.front_guidance_gate_max > 0
            elif self.learnable_front_guidance:
                if hasattr(self, "front_guidance_alpha"):
                    raw_alpha = self._front_guidance_raw_alpha(
                        device=q.device
                    )
                else:
                    raw_alpha = torch.as_tensor(
                        self.front_guidance_alpha_init,
                        device=q.device,
                        dtype=torch.float32,
                    )
                alpha = self._straight_through_clamp_front_guidance_alpha(
                    raw_alpha
                ).to(device=q.device, dtype=q.dtype)
                progress = max(0.0, min(1.0, float(train_progress)))
                alpha = alpha * progress
                should_compute_guided = self.front_guidance_alpha_max > 0
            else:
                progress = max(0.0, min(1.0, float(train_progress)))
                alpha = self.target_guidance_scale * progress
                should_compute_guided = alpha > 0

            if should_compute_guided:
                if self.front_guidance_mode == "soft_bias":
                    min_area = max(0.0, min(1.0, float(self.front_guidance_min_area)))
                    max_area = max(0.0, min(1.0, float(self.front_guidance_max_area)))
                    if max_area < min_area:
                        max_area = min_area

                    area = ref_mask.mean(dim=1, keepdim=True)
                    valid_area = ((area >= min_area) & (area <= max_area)).to(
                        dtype=ref_mask.dtype
                    )
                    if (
                        self.training
                        and self.front_guidance_dropout_prob > 0
                    ):
                        keep_prob = max(
                            0.0,
                            min(1.0, 1.0 - float(self.front_guidance_dropout_prob)),
                        )
                        if keep_prob <= 0:
                            valid_area = torch.zeros_like(valid_area)
                        else:
                            keep_mask = (
                                torch.rand_like(valid_area) < keep_prob
                            ).to(dtype=valid_area.dtype)
                            valid_area = valid_area * keep_mask
                    centered_mask = ref_mask - area
                    mask_scale = centered_mask.abs().amax(
                        dim=1, keepdim=True
                    ).clamp_min(1e-6)
                    centered_mask = (centered_mask / mask_scale) * valid_area

                    if self.adaptive_front_guidance:
                        adaptive_gate = self._front_guidance_adaptive_gate(
                            area=area,
                            valid_area=valid_area,
                            train_progress=train_progress,
                            device=q.device,
                            dtype=q.dtype,
                        )
                        alpha_tensor = adaptive_gate
                    elif torch.is_tensor(alpha):
                        alpha_tensor = alpha.to(device=q.device, dtype=q.dtype)
                    else:
                        alpha_tensor = torch.tensor(
                            float(alpha), device=q.device, dtype=q.dtype
                        )
                    key_bias = (
                        centered_mask[:, None, :, None].to(dtype=q.dtype)
                        * alpha_tensor
                        * float(self.boost_strength)
                    )
                    # Add a per-key logit bias without passing attn_mask to
                    # SDPA. This avoids fragile masked FlashAttention backward
                    # kernels while preserving the original attention scale.
                    q_extra = torch.ones(
                        (*q.shape[:-1], 1),
                        device=q.device,
                        dtype=q.dtype,
                    )
                    k_extra = (
                        key_bias / self.scale
                    ).expand(k.shape[0], k.shape[1], k.shape[2], 1)
                    q_guided = torch.cat([q, q_extra], dim=-1).contiguous()
                    k_guided = torch.cat([k, k_extra], dim=-1).contiguous()
                    out = F.scaled_dot_product_attention(
                        q_guided,
                        k_guided,
                        v.contiguous(),
                        dropout_p=0.0,
                        is_causal=False,
                        scale=self.scale,
                    )
                elif self.front_guidance_mode == "residual_gate":
                    min_area = max(0.0, min(1.0, float(self.front_guidance_min_area)))
                    max_area = max(0.0, min(1.0, float(self.front_guidance_max_area)))
                    if max_area < min_area:
                        max_area = min_area

                    area = ref_mask.mean(dim=1, keepdim=True)
                    valid_area = ((area >= min_area) & (area <= max_area)).to(
                        dtype=ref_mask.dtype
                    )
                    if (
                        self.training
                        and self.front_guidance_dropout_prob > 0
                    ):
                        keep_prob = max(
                            0.0,
                            min(1.0, 1.0 - float(self.front_guidance_dropout_prob)),
                        )
                        if keep_prob <= 0:
                            valid_area = torch.zeros_like(valid_area)
                        else:
                            keep_mask = (
                                torch.rand_like(valid_area) < keep_prob
                            ).to(dtype=valid_area.dtype)
                            valid_area = valid_area * keep_mask

                    centered_mask = ref_mask - area
                    mask_scale = centered_mask.abs().amax(
                        dim=1, keepdim=True
                    ).clamp_min(1e-6)
                    centered_mask = (centered_mask / mask_scale) * valid_area

                    if self.adaptive_front_guidance:
                        adaptive_gate = self._front_guidance_adaptive_gate(
                            area=area,
                            valid_area=valid_area,
                            train_progress=train_progress,
                            device=q.device,
                            dtype=q.dtype,
                        )
                        alpha_tensor = adaptive_gate
                    elif torch.is_tensor(alpha):
                        alpha_tensor = alpha.to(device=q.device, dtype=q.dtype)
                    else:
                        alpha_tensor = torch.tensor(
                            float(alpha), device=q.device, dtype=q.dtype
                        )

                    # Preserve the original attention distribution. Front only
                    # scales value residuals, so a noisy mask cannot redirect
                    # queries to different keys.
                    value_delta = v * (
                        centered_mask[:, None, :, None].to(dtype=q.dtype)
                        * float(self.boost_strength)
                    )
                    out_delta = F.scaled_dot_product_attention(
                        q.contiguous(),
                        k.contiguous(),
                        value_delta.contiguous(),
                        dropout_p=0.0,
                        is_causal=False,
                    )
                    out = out_global + alpha_tensor * out_delta
                elif self.front_guidance_mode == "foreground_memory":
                    if not hasattr(self, "front_memory_out"):
                        self.init_front_memory()
                    (
                        memory_k,
                        memory_v,
                        quality,
                        memory_valid,
                    ) = self._front_memory_tokens(
                        k=k,
                        v=v,
                        ref_mask=ref_mask,
                        ref_token_shape=ref_token_shape,
                    )
                    quality = self._apply_front_memory_dropout(
                        quality=quality,
                        memory_valid=memory_valid,
                    )
                    gate = self._front_memory_gate(
                        quality=quality,
                        device=q.device,
                        dtype=q.dtype,
                    )
                    if memory_k is None or memory_v is None:
                        out = out_global
                    else:
                        front_logits = torch.matmul(
                            q.detach().float(),
                            memory_k.detach().float().transpose(-1, -2),
                        ) * self.scale
                        front_affinity = torch.sigmoid(
                            front_logits.mean(dim=1).amax(dim=-1)
                        ).view(b, seq_len, 1)
                        min_alignment = float(self.front_memory_min_alignment)
                        if min_alignment >= 1.0:
                            affinity_quality = torch.zeros_like(front_affinity)
                        else:
                            affinity_quality = (
                                (front_affinity - min_alignment)
                                / max(1e-6, 1.0 - min_alignment)
                            ).clamp(0.0, 1.0)
                        self._last_front_memory_alignment = (
                            front_affinity.mean(dim=1).detach().float()
                        )
                        out_global_flat = (
                            out_global.permute(0, 2, 1, 3)
                            .reshape(b, seq_len, -1)
                        )
                        memory_attn = F.scaled_dot_product_attention(
                            q.detach().contiguous(),
                            memory_k.detach().contiguous(),
                            memory_v.detach().contiguous(),
                            dropout_p=0.0,
                            is_causal=False,
                        )
                        memory_attn_flat = (
                            memory_attn.permute(0, 2, 1, 3)
                            .reshape(b, seq_len, -1)
                        )
                        memory_delta = memory_attn_flat - out_global_flat.detach()
                        out_memory = self.front_memory_out(memory_delta)
                        query_gate = None
                        query_mask_power = max(
                            0.0,
                            float(getattr(self, "front_memory_query_mask_power", 0.0)),
                        )
                        if query_mask_power > 0.0:
                            query_mask = ref_mask
                            if query_mask.shape[1] < seq_len:
                                query_mask = F.pad(
                                    query_mask,
                                    (0, seq_len - query_mask.shape[1]),
                                )
                            elif query_mask.shape[1] > seq_len:
                                query_mask = query_mask[:, :seq_len]
                            query_gate = (
                                query_mask.to(
                                    device=out_memory.device,
                                    dtype=out_memory.dtype,
                                )
                                .clamp(0.0, 1.0)
                                .view(b, seq_len, 1)
                            )
                            if query_mask_power != 1.0:
                                query_gate = query_gate.pow(query_mask_power)
                            query_floor = max(
                                0.0,
                                min(
                                    1.0,
                                    float(
                                        getattr(
                                            self,
                                            "front_memory_query_mask_floor",
                                            0.0,
                                        )
                                    ),
                                ),
                            )
                            if query_floor > 0.0:
                                query_gate = query_floor + (
                                    1.0 - query_floor
                                ) * query_gate
                            self._last_front_memory_query_gate = (
                                query_gate.detach().float().mean(dim=1)
                            )
                        else:
                            self._last_front_memory_query_gate = None
                        memory_gate = gate * affinity_quality.mean(dim=1).to(
                            device=gate.device,
                            dtype=gate.dtype,
                        )
                        memory_scale = (
                            affinity_quality.to(dtype=out_memory.dtype)
                            * memory_gate[:, None, :].to(dtype=out_memory.dtype)
                            * float(self.boost_strength)
                        )
                        grad_multiplier = max(
                            1.0,
                            float(
                                getattr(
                                    self,
                                    "front_memory_train_gate_gradient_multiplier",
                                    1.0,
                                )
                            ),
                        )
                        if self.training and grad_multiplier > 1.0:
                            out_forward = out_memory * memory_scale
                            out_grad = out_memory * (memory_scale * grad_multiplier)
                            out_memory = out_grad + (out_forward - out_grad).detach()
                            self._last_front_memory_train_gate_gradient_multiplier = (
                                out_memory.new_full((b, 1), grad_multiplier)
                                .detach()
                                .float()
                            )
                        else:
                            out_memory = out_memory * memory_scale
                            self._last_front_memory_train_gate_gradient_multiplier = None
                        if query_gate is not None:
                            out_memory = out_memory * query_gate.to(
                                dtype=out_memory.dtype
                            )
                        residual_ratio_max = float(
                            getattr(self, "front_memory_residual_ratio_max", -1.0)
                        )
                        if residual_ratio_max >= 0.0:
                            memory_norm = (
                                out_memory.detach()
                                .float()
                                .norm(dim=(1, 2))
                                .view(b, 1, 1)
                            )
                            max_memory_norm = (
                                out_global_flat.detach()
                                .float()
                                .norm(dim=(1, 2))
                                .view(b, 1, 1)
                                * residual_ratio_max
                            )
                            memory_scale = (
                                max_memory_norm / memory_norm.clamp_min(1e-12)
                            ).clamp(max=1.0)
                            out_memory = out_memory * memory_scale.to(
                                dtype=out_memory.dtype
                            )
                        out = out_global
                        self._last_front_memory_residual_ratio = (
                            out_memory.detach().float().norm(dim=(1, 2))
                            / out_global_flat.detach()
                            .float()
                            .norm(dim=(1, 2))
                            .clamp_min(1e-12)
                        )
                        out_extra = out_memory
                elif self.front_guidance_mode == "blend":
                    if self.adaptive_front_guidance:
                        area = ref_mask.mean(dim=1, keepdim=True)
                        valid_area = torch.ones_like(area)
                        adaptive_gate = self._front_guidance_adaptive_gate(
                            area=area,
                            valid_area=valid_area,
                            train_progress=train_progress,
                            device=q.device,
                            dtype=q.dtype,
                        )
                        alpha_tensor = adaptive_gate
                    elif torch.is_tensor(alpha):
                        alpha_tensor = alpha.to(device=q.device, dtype=q.dtype)
                    else:
                        alpha_tensor = torch.tensor(
                            float(alpha), device=q.device, dtype=q.dtype
                        )
                    bias_guided = ref_mask[:, None, None, :].to(
                        device=q.device, dtype=q.dtype
                    ) * self.boost_strength
                    out_guided = F.scaled_dot_product_attention(
                        q.contiguous(),
                        k.contiguous(),
                        v.contiguous(),
                        attn_mask=bias_guided,
                        dropout_p=0.0,
                        is_causal=False,
                    )
                    out = out_global + alpha_tensor * (out_guided - out_global)
                else:
                    raise ValueError(
                        f"Unknown front_guidance_mode: {self.front_guidance_mode}"
                    )
            else:
                out = out_global
        else:
            out = out_global

        # Reshape and permute to prepare for final linear transformation
        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)
        if out_extra is not None:
            out = out + out_extra.to(dtype=out.dtype)

        return self.to_out(out)

    def init_adaptive_front_guidance(self):
        for name in (
            "front_guidance_gate_bias",
            "front_guidance_area_weight",
            "front_guidance_progress_weight",
        ):
            if hasattr(self, name):
                getattr(self, name).requires_grad_(self.adaptive_front_guidance)
                continue
            setattr(
                self,
                name,
                nn.Parameter(
                    torch.zeros((), dtype=torch.float32),
                    requires_grad=self.adaptive_front_guidance,
                ),
            )

    @staticmethod
    def _front_gate_bias_from_init(gate_init: float, gate_max: float) -> float:
        if gate_max <= 0:
            return -20.0
        prob = float(gate_init) / float(gate_max)
        prob = min(max(prob, 1e-4), 1.0 - 1e-4)
        return math.log(prob / (1.0 - prob))

    def init_front_memory(self):
        requires_grad = self.front_guidance_mode == "foreground_memory"
        if hasattr(self, "front_memory_out"):
            self.front_memory_out.requires_grad_(requires_grad)
            if hasattr(self, "front_memory_gate_bias"):
                self.front_memory_gate_bias.requires_grad_(requires_grad)
                self.front_memory_gate_bias.data = self.front_memory_gate_bias.data.float()
            return

        self.front_memory_out = nn.Linear(
            self.inner_dim,
            self.inner_dim,
            bias=False,
        )
        nn.init.zeros_(self.front_memory_out.weight)
        if hasattr(self, "to_out"):
            self.front_memory_out.to(
                device=self.to_out.weight.device,
                dtype=self.to_out.weight.dtype,
            )
        self.front_memory_gate_bias = nn.Parameter(
            torch.tensor(
                self._front_gate_bias_from_init(
                    self.front_guidance_gate_init,
                    self.front_guidance_gate_max,
                ),
                dtype=torch.float32,
            ),
            requires_grad=requires_grad,
        )
        self.front_memory_out.requires_grad_(requires_grad)

    def _front_memory_gate(self, *, quality, device, dtype):
        if not hasattr(self, "front_memory_gate_bias"):
            self.init_front_memory()
        gate_max = max(0.0, float(self.front_guidance_gate_max))
        gate_floor = max(
            0.0,
            min(gate_max, float(getattr(self, "front_guidance_gate_floor", 0.0))),
        )
        gate_span = max(0.0, gate_max - gate_floor)
        gate_param = gate_floor + gate_span * torch.sigmoid(
            self.front_memory_gate_bias.to(device=device, dtype=torch.float32)
        )
        gate = gate_param * quality.to(device=device, dtype=torch.float32)
        self._last_front_guidance_gate = gate.detach().float()
        self._last_front_guidance_gate_for_loss = gate.float().mean()
        return gate.to(device=device, dtype=dtype)

    def _apply_front_memory_dropout(self, *, quality, memory_valid):
        if (
            self.training
            and self.front_guidance_dropout_prob > 0
        ):
            keep_prob = max(
                0.0,
                min(1.0, 1.0 - float(self.front_guidance_dropout_prob)),
            )
            if keep_prob <= 0:
                keep_mask = torch.zeros_like(quality)
            else:
                keep_mask = (torch.rand_like(quality) < keep_prob).to(
                    dtype=quality.dtype
                )
            quality = quality * keep_mask
            memory_valid = memory_valid * keep_mask
        self._last_front_memory_quality = quality.detach().float()
        self._last_front_memory_enabled = memory_valid.detach().float()
        self._last_front_guidance_valid_area = memory_valid.detach().float()
        return quality

    def _front_memory_tokens(self, *, k, v, ref_mask, ref_token_shape=None):
        batch, heads, key_len, dim = k.shape
        mask = ref_mask.to(device=k.device, dtype=torch.float32).clamp(0.0, 1.0)
        min_area = max(0.0, min(1.0, float(self.front_guidance_min_area)))
        max_area = max(0.0, min(1.0, float(self.front_guidance_max_area)))
        if max_area < min_area:
            max_area = min_area

        sample_area = mask.mean(dim=1, keepdim=True)
        sample_valid = ((sample_area >= min_area) & (sample_area <= max_area)).to(
            dtype=mask.dtype
        )

        frames = 1
        spatial_tokens = key_len
        spatial_height = None
        spatial_width = None
        if ref_token_shape is not None and len(ref_token_shape) == 3:
            cand_frames, cand_height, cand_width = [int(value) for value in ref_token_shape]
            cand_spatial = cand_height * cand_width
            if cand_frames > 0 and cand_spatial > 0 and cand_frames * cand_spatial == key_len:
                frames = cand_frames
                spatial_tokens = cand_spatial
                spatial_height = cand_height
                spatial_width = cand_width

        mask_view = mask.reshape(batch, frames, spatial_tokens)
        k_view = k.reshape(batch, heads, frames, spatial_tokens, dim)
        v_view = v.reshape(batch, heads, frames, spatial_tokens, dim)

        frame_area = mask_view.mean(dim=2)
        frame_valid = ((frame_area >= min_area) & (frame_area <= max_area)).to(
            dtype=mask.dtype
        )
        valid_frame_ratio = frame_valid.mean(dim=1, keepdim=True)
        area_std = frame_area.float().std(dim=1, unbiased=False, keepdim=True).to(
            dtype=mask.dtype
        )
        if frames > 1:
            max_area_jump = (
                (frame_area[:, 1:] - frame_area[:, :-1])
                .abs()
                .max(dim=1, keepdim=True)
                .values
            )
        else:
            max_area_jump = torch.zeros_like(valid_frame_ratio)

        high_threshold = max(
            0.0,
            min(1.0, float(self.front_memory_high_threshold)),
        )
        high_weights = mask_view * (mask_view >= high_threshold).to(dtype=mask.dtype)
        fallback_weights = mask_view.square()
        has_high = (high_weights.sum(dim=2, keepdim=True) > 1e-6).to(dtype=mask.dtype)
        core_weights = high_weights + (1.0 - has_high) * fallback_weights

        boundary_weights = (
            4.0
            * mask_view
            * (1.0 - mask_view)
            * (mask_view > 0.02).to(dtype=mask.dtype)
            * (mask_view < high_threshold).to(dtype=mask.dtype)
        ).clamp(0.0, 1.0)

        core_weights = core_weights * frame_valid[:, :, None]
        boundary_weights = boundary_weights * frame_valid[:, :, None]

        def _pool_tokens(weights):
            weights = weights[:, None, :, :, None].to(device=k.device, dtype=k.dtype)
            denom = weights.sum(dim=3).clamp_min(1e-6)
            pooled_k = (k_view * weights).sum(dim=3) / denom
            pooled_v = (v_view * weights).sum(dim=3) / denom
            return pooled_k, pooled_v

        core_k, core_v = _pool_tokens(core_weights)
        boundary_k, boundary_v = _pool_tokens(boundary_weights)
        memory_k = torch.cat([core_k, boundary_k], dim=2)
        memory_v = torch.cat([core_v, boundary_v], dim=2)

        core_area = (core_weights > 1e-6).to(dtype=mask.dtype).mean(dim=(1, 2), keepdim=True)
        boundary_area = (boundary_weights > 1e-6).to(dtype=mask.dtype).mean(dim=(1, 2), keepdim=True)
        high_mass_ratio = high_weights.sum(dim=(1, 2), keepdim=True) / mask_view.sum(
            dim=(1, 2), keepdim=True
        ).clamp_min(1e-6)
        high_mass_ratio = high_mass_ratio.view(batch, 1)
        if spatial_height is not None and spatial_width is not None:
            mask_grid = mask_view.reshape(batch, frames, spatial_height, spatial_width)
            if spatial_width > 1:
                noise_x = (
                    mask_grid[:, :, :, 1:] - mask_grid[:, :, :, :-1]
                ).abs().mean(dim=(1, 2, 3))
            else:
                noise_x = torch.zeros(batch, device=mask.device, dtype=mask.dtype)
            if spatial_height > 1:
                noise_y = (
                    mask_grid[:, :, 1:, :] - mask_grid[:, :, :-1, :]
                ).abs().mean(dim=(1, 2, 3))
            else:
                noise_y = torch.zeros(batch, device=mask.device, dtype=mask.dtype)
            noise_score = 0.5 * (noise_x + noise_y)
        else:
            noise_score = torch.zeros(batch, device=mask.device, dtype=mask.dtype)
        high_area = (mask_view >= high_threshold).to(dtype=mask.dtype).mean(dim=2)
        soft_area = (
            ((mask_view > 0.02) & (mask_view < high_threshold))
            .to(dtype=mask.dtype)
            .mean(dim=2)
        )
        high_active_ratio = (
            high_area / (high_area + soft_area).clamp_min(1e-8)
        ).mean(dim=1, keepdim=True)
        noise_threshold = max(1e-6, float(self.front_memory_noise_threshold))
        smooth_quality = (1.0 - noise_score.view(batch, 1) / noise_threshold).clamp(
            0.0,
            1.0,
        )
        enabled = (
            (
                (core_weights.sum(dim=2) > 1e-6)
                | (boundary_weights.sum(dim=2) > 1e-6)
            )
            .to(dtype=mask.dtype)
            .mean(dim=1, keepdim=True)
        )
        quality_threshold = max(1e-6, float(self.front_memory_quality_threshold))
        area_quality = (sample_area / quality_threshold).clamp(0.0, 1.0)
        quality = (
            sample_valid
            * enabled
            * smooth_quality
            * (0.5 * area_quality + 0.5 * high_mass_ratio)
        )
        sample_gate = torch.ones_like(quality)
        min_area_std = float(getattr(self, "front_memory_min_area_std", -1.0))
        if min_area_std >= 0.0:
            sample_gate = sample_gate * (area_std >= min_area_std).to(dtype=mask.dtype)
        max_area_std = float(getattr(self, "front_memory_max_area_std", -1.0))
        if max_area_std >= 0.0:
            sample_gate = sample_gate * (area_std <= max_area_std).to(dtype=mask.dtype)
        min_valid_frame_ratio = float(
            getattr(self, "front_memory_min_valid_frame_ratio", 0.0)
        )
        if min_valid_frame_ratio > 0.0:
            sample_gate = sample_gate * (
                valid_frame_ratio >= min_valid_frame_ratio
            ).to(dtype=mask.dtype)
        max_area_jump_limit = float(getattr(self, "front_memory_max_area_jump", -1.0))
        if max_area_jump_limit >= 0.0:
            sample_gate = sample_gate * (
                max_area_jump <= max_area_jump_limit
            ).to(dtype=mask.dtype)
        max_spatial_noise = float(
            getattr(self, "front_memory_max_spatial_noise", -1.0)
        )
        if max_spatial_noise >= 0.0:
            sample_gate = sample_gate * (
                noise_score.view(batch, 1) <= max_spatial_noise
            ).to(dtype=mask.dtype)
        min_high_active_ratio = float(
            getattr(self, "front_memory_min_high_active_ratio", 0.0)
        )
        if min_high_active_ratio > 0.0:
            sample_gate = sample_gate * (
                high_active_ratio >= min_high_active_ratio
            ).to(dtype=mask.dtype)
        quality = quality * sample_gate
        quality = quality.clamp(0.0, 1.0).view(batch, 1)
        quality_power = max(1.0, float(getattr(self, "front_memory_quality_power", 1.0)))
        if quality_power != 1.0:
            quality = quality.pow(quality_power)
        memory_valid = (
            sample_valid * enabled * sample_gate
        ).clamp(0.0, 1.0).view(batch, 1)

        self._last_front_guidance_area = sample_area.detach().float()
        self._last_front_memory_noise = noise_score.view(batch, 1).detach().float()
        self._last_front_memory_sample_gate = sample_gate.detach().float()
        self._last_front_memory_area_std = area_std.detach().float()
        self._last_front_memory_max_area_jump = max_area_jump.detach().float()
        self._last_front_memory_valid_frame_ratio = valid_frame_ratio.detach().float()
        self._last_front_memory_high_active_ratio = high_active_ratio.detach().float()
        self._last_front_memory_core_area = core_area.view(batch, 1).detach().float()
        self._last_front_memory_boundary_area = boundary_area.view(batch, 1).detach().float()
        return memory_k, memory_v, quality, memory_valid

    def init_front_guidance_alpha(self):
        if hasattr(self, "front_guidance_alpha"):
            self.front_guidance_alpha.requires_grad_(self.learnable_front_guidance)
            return
        # Store a delta around zero instead of the absolute alpha. Under bf16
        # training, updates near 0.001 are smaller than one representable bf16
        # step; a zero-centered delta can still move with small learning rates.
        self.front_guidance_alpha = nn.Parameter(
            torch.zeros((), dtype=torch.float32),
            requires_grad=self.learnable_front_guidance,
        )

    def _front_guidance_raw_alpha(self, device=None):
        if device is None:
            device = self.front_guidance_alpha.device
        alpha_delta = self.front_guidance_alpha.to(
            device=device,
            dtype=torch.float32,
        )
        alpha_base = torch.as_tensor(
            self.front_guidance_alpha_init,
            device=device,
            dtype=torch.float32,
        )
        return alpha_base + alpha_delta

    def _straight_through_clamp_front_guidance_alpha(self, raw_alpha):
        clamped_alpha = torch.clamp(
            raw_alpha,
            min=0.0,
            max=max(0.0, self.front_guidance_alpha_max),
        )
        return raw_alpha + (clamped_alpha - raw_alpha).detach()

    def _straight_through_clamp_front_guidance_gate(self, raw_gate):
        gate_max = max(0.0, float(self.front_guidance_gate_max))
        gate_floor = max(
            0.0,
            min(gate_max, float(getattr(self, "front_guidance_gate_floor", 0.0))),
        )
        clamped_gate = torch.clamp(
            raw_gate,
            min=gate_floor,
            max=gate_max,
        )
        return raw_gate + (clamped_gate - raw_gate).detach()

    def _front_guidance_adaptive_gate(
        self,
        *,
        area,
        valid_area,
        train_progress: float,
        device,
        dtype,
    ):
        if not hasattr(self, "front_guidance_gate_bias"):
            self.init_adaptive_front_guidance()

        progress = max(0.0, min(1.0, float(train_progress)))
        progress_tensor = torch.as_tensor(
            progress,
            device=device,
            dtype=torch.float32,
        )
        gate_base = torch.as_tensor(
            self.front_guidance_gate_init,
            device=device,
            dtype=torch.float32,
        )
        raw_gate = (
            gate_base
            + self.front_guidance_gate_bias.to(device=device, dtype=torch.float32)
            + self.front_guidance_area_weight.to(device=device, dtype=torch.float32)
            * (area.to(device=device, dtype=torch.float32) - 0.15)
            + self.front_guidance_progress_weight.to(device=device, dtype=torch.float32)
            * progress_tensor
        )
        gate = self._straight_through_clamp_front_guidance_gate(raw_gate)
        gate = gate * valid_area.to(device=device, dtype=torch.float32)
        self._last_front_guidance_gate = gate.detach().float()
        self._last_front_guidance_gate_for_loss = gate.float().mean()
        self._last_front_guidance_area = area.detach().float()
        self._last_front_guidance_valid_area = valid_area.detach().float()
        return gate[:, None, None, :].to(device=device, dtype=dtype)


class CrossTransformer3DModel(ModelMixin, ConfigMixin):
    """
    A Transformer model for video-like data in [CogVideoX](https://github.com/THUDM/CogVideo).

    Parameters:
        num_attention_heads (`int`, defaults to `30`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `16`):
            The number of channels in the output.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        attention_bias (`bool`, defaults to `True`):
            Whether or not to use bias in the attention projection layers.
        sample_width (`int`, defaults to `90`):
            The width of the input latents.
        sample_height (`int`, defaults to `60`):
            The height of the input latents.
        sample_frames (`int`, defaults to `49`):
            The number of frames in the input latents. Note that this parameter was incorrectly initialized to 49
            instead of 13 because CogVideoX processed 13 latent frames at once in its default and recommended settings,
            but cannot be changed to the correct value to ensure backwards compatibility. To create a transformer with
            K latent frames, the correct value to pass here would be: ((K - 1) * temporal_compression_ratio + 1).
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        temporal_compression_ratio (`int`, defaults to `4`):
            The compression ratio across the temporal dimension. See documentation for `sample_frames`.
        max_text_seq_length (`int`, defaults to `226`):
            The maximum sequence length of the input text embeddings.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        timestep_activation_fn (`str`, defaults to `"silu"`):
            Activation function to use when generating the timestep embeddings.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether or not to use elementwise affine in normalization layers.
        norm_eps (`float`, defaults to `1e-5`):
            The epsilon value to use in normalization layers.
        spatial_interpolation_scale (`float`, defaults to `1.875`):
            Scaling factor to apply in 3D positional embeddings across spatial dimensions.
        temporal_interpolation_scale (`float`, defaults to `1.0`):
            Scaling factor to apply in 3D positional embeddings across temporal dimensions.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        add_noise_in_inpaint_model: bool = False,
        is_train_cross: bool = False,
        cross_attn_in_channels: int = 16,
        cross_attn_interval: int = 2,
        cross_attn_dim_head: int = 128,
        cross_attn_num_heads: int = 16,
        enable_front_guidance: bool = False,
        front_guidance_target_scale: float = 0.6,
        front_guidance_boost_strength: float = 5.0,
        front_guidance_mode: str = "blend",
        front_guidance_min_area: float = 0.0,
        front_guidance_max_area: float = 1.0,
        front_guidance_dropout_prob: float = 0.0,
        learnable_front_guidance: bool = False,
        front_guidance_alpha_max: float = 0.15,
        front_guidance_alpha_init: float = 1e-3,
        adaptive_front_guidance: bool = False,
        front_guidance_gate_max: float = 0.03,
        front_guidance_gate_init: float = 0.0,
        front_guidance_gate_floor: float = 0.0,
        front_memory_high_threshold: float = 0.55,
        front_memory_quality_threshold: float = 0.01,
        front_memory_noise_threshold: float = 0.08,
        front_memory_min_alignment: float = 0.0,
        front_memory_quality_power: float = 1.0,
        front_memory_residual_ratio_max: float = -1.0,
        front_memory_query_mask_power: float = 0.0,
        front_memory_query_mask_floor: float = 0.0,
        front_memory_train_gate_gradient_multiplier: float = 1.0,
        front_memory_max_area_std: float = -1.0,
        front_memory_min_area_std: float = -1.0,
        front_memory_min_valid_frame_ratio: float = 0.0,
        front_memory_max_area_jump: float = -1.0,
        front_memory_max_spatial_noise: float = -1.0,
        front_memory_min_high_active_ratio: float = 0.0,
        enable_front_condition: bool = False,
        front_condition_use_boundary: bool = True,
        front_condition_strength_init: float = 1.0,
        front_condition_gate_init: float = 0.05,
        front_condition_gate_max: float = 0.25,
        front_condition_video_scale_init: float = 0.0,
        front_condition_front_scale_init: float = 0.0,
        front_condition_boundary_scale_init: float = 0.0,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        post_patch_height = sample_height // patch_size
        post_patch_width = sample_width // patch_size
        post_time_compression_frames = (
            sample_frames - 1
        ) // temporal_compression_ratio + 1
        self.num_patches = (
            post_patch_height * post_patch_width * post_time_compression_frames
        )
        self.post_patch_height = post_patch_height
        self.post_patch_width = post_patch_width
        self.post_time_compression_frames = post_time_compression_frames
        self.patch_size = patch_size
        self.inner_dim = inner_dim

        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size, in_channels, inner_dim, text_embed_dim, bias=True
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. 3D positional embeddings
        spatial_pos_embedding = _get_3d_sincos_pos_embed_tensor(
            inner_dim,
            (post_patch_width, post_patch_height),
            post_time_compression_frames,
            spatial_interpolation_scale,
            temporal_interpolation_scale,
        ).flatten(0, 1)
        pos_embedding = torch.zeros(
            1, max_text_seq_length + self.num_patches, inner_dim, requires_grad=False
        )
        pos_embedding.data[:, max_text_seq_length:].copy_(spatial_pos_embedding)
        self.register_buffer("pos_embedding", pos_embedding, persistent=False)

        # 3. Time embeddings
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(
            inner_dim, time_embed_dim, timestep_activation_fn
        )

        # 4. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 5. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)

        self.gradient_checkpointing = False

        self.is_train_cross = is_train_cross
        self.enable_front_guidance = bool(enable_front_guidance)
        self.front_guidance_target_scale = float(front_guidance_target_scale)
        self.front_guidance_boost_strength = float(front_guidance_boost_strength)
        self.front_guidance_mode = str(front_guidance_mode)
        self.front_guidance_min_area = float(front_guidance_min_area)
        self.front_guidance_max_area = float(front_guidance_max_area)
        self.front_guidance_dropout_prob = float(front_guidance_dropout_prob)
        self.learnable_front_guidance = bool(learnable_front_guidance)
        self.front_guidance_alpha_max = float(front_guidance_alpha_max)
        self.front_guidance_alpha_init = float(front_guidance_alpha_init)
        self.adaptive_front_guidance = bool(adaptive_front_guidance)
        self.front_guidance_gate_max = float(front_guidance_gate_max)
        self.front_guidance_gate_init = float(front_guidance_gate_init)
        self.front_guidance_gate_floor = float(front_guidance_gate_floor)
        self.front_memory_high_threshold = float(front_memory_high_threshold)
        self.front_memory_quality_threshold = float(front_memory_quality_threshold)
        self.front_memory_noise_threshold = float(front_memory_noise_threshold)
        self.front_memory_min_alignment = float(front_memory_min_alignment)
        self.front_memory_quality_power = float(front_memory_quality_power)
        self.front_memory_residual_ratio_max = float(front_memory_residual_ratio_max)
        self.front_memory_query_mask_power = float(front_memory_query_mask_power)
        self.front_memory_query_mask_floor = float(front_memory_query_mask_floor)
        self.front_memory_train_gate_gradient_multiplier = float(
            front_memory_train_gate_gradient_multiplier
        )
        self.front_memory_max_area_std = float(front_memory_max_area_std)
        self.front_memory_min_area_std = float(front_memory_min_area_std)
        self.front_memory_min_valid_frame_ratio = float(
            front_memory_min_valid_frame_ratio
        )
        self.front_memory_max_area_jump = float(front_memory_max_area_jump)
        self.front_memory_max_spatial_noise = float(front_memory_max_spatial_noise)
        self.front_memory_min_high_active_ratio = float(
            front_memory_min_high_active_ratio
        )
        self.enable_front_condition = bool(enable_front_condition)
        self.front_condition_use_boundary = bool(front_condition_use_boundary)
        self.front_condition_strength_init = float(front_condition_strength_init)
        self.front_condition_gate_init = float(front_condition_gate_init)
        self.front_condition_gate_max = float(front_condition_gate_max)
        self.front_condition_video_scale_init = float(front_condition_video_scale_init)
        self.front_condition_front_scale_init = float(front_condition_front_scale_init)
        self.front_condition_boundary_scale_init = float(front_condition_boundary_scale_init)
        if self.enable_front_condition:
            self._init_front_condition_adapter()
        if is_train_cross:
            # cross configs
            self.cross_attn_interval = cross_attn_interval
            self.num_cross_attn = num_layers // cross_attn_interval
            self.cross_attn_dim_head = cross_attn_dim_head
            self.cross_attn_num_heads = cross_attn_num_heads
            self.cross_attn_kv_dim = None
            self.ref_patch_embed = RefPatchEmbed(
                patch_size, cross_attn_in_channels, inner_dim, bias=True
            )
            self._init_cross_inputs()

    def _init_front_condition_adapter(self):
        if hasattr(self, "front_condition_adapter"):
            expected_in_channels = 2 if self.front_condition_use_boundary else 1
            existing_proj = getattr(self.front_condition_adapter, "delta_proj", None)
            if existing_proj is None or existing_proj.in_channels != expected_in_channels:
                del self.front_condition_adapter
            else:
                self.front_condition_adapter.use_boundary = self.front_condition_use_boundary
                self.front_condition_adapter.gate_init = self.front_condition_gate_init
                self.front_condition_adapter.gate_max = self.front_condition_gate_max
                for scalar_name in (
                    "strength",
                    "gate_bias",
                    "video_scale",
                    "front_scale",
                    "boundary_scale",
                ):
                    scalar_param = getattr(self.front_condition_adapter, scalar_name)
                    scalar_param.data = scalar_param.data.float()
                self.front_condition_adapter.strength.requires_grad_(self.enable_front_condition)
                self.front_condition_adapter.gate_bias.requires_grad_(self.enable_front_condition)
                self.front_condition_adapter.video_scale.requires_grad_(self.enable_front_condition)
                self.front_condition_adapter.front_scale.requires_grad_(self.enable_front_condition)
                self.front_condition_adapter.boundary_scale.requires_grad_(self.enable_front_condition)
                for param in self.front_condition_adapter.delta_proj.parameters():
                    param.requires_grad_(self.enable_front_condition)
                return
        self.front_condition_adapter = FrontConditionAdapter(
            patch_size=self.patch_size,
            embed_dim=self.inner_dim if hasattr(self, "inner_dim") else self.config.num_attention_heads * self.config.attention_head_dim,
            use_boundary=self.front_condition_use_boundary,
            strength_init=self.front_condition_strength_init,
            gate_init=self.front_condition_gate_init,
            gate_max=self.front_condition_gate_max,
            video_scale_init=self.front_condition_video_scale_init,
            front_scale_init=self.front_condition_front_scale_init,
            boundary_scale_init=self.front_condition_boundary_scale_init,
        ).to(self.device, dtype=self.dtype)
        for scalar_name in (
            "strength",
            "gate_bias",
            "video_scale",
            "front_scale",
            "boundary_scale",
        ):
            scalar_param = getattr(self.front_condition_adapter, scalar_name)
            scalar_param.data = scalar_param.data.float()

    def _init_cross_inputs(self):
        device = self.device
        weight_dtype = self.dtype
        self.perceiver_cross_attention = nn.ModuleList(
            [
                PerceiverCrossAttention(
                    dim=self.inner_dim,
                    dim_head=self.cross_attn_dim_head,
                    heads=self.cross_attn_num_heads,
                    kv_dim=self.cross_attn_kv_dim,
                    enable_front_guidance=self.enable_front_guidance,
                    target_guidance_scale=self.front_guidance_target_scale,
                    boost_strength=self.front_guidance_boost_strength,
                    front_guidance_mode=self.front_guidance_mode,
                    front_guidance_min_area=self.front_guidance_min_area,
                    front_guidance_max_area=self.front_guidance_max_area,
                    front_guidance_dropout_prob=self.front_guidance_dropout_prob,
                    learnable_front_guidance=self.learnable_front_guidance,
                    front_guidance_alpha_max=self.front_guidance_alpha_max,
                    front_guidance_alpha_init=self.front_guidance_alpha_init,
                    adaptive_front_guidance=self.adaptive_front_guidance,
                    front_guidance_gate_max=self.front_guidance_gate_max,
                    front_guidance_gate_init=self.front_guidance_gate_init,
                    front_guidance_gate_floor=self.front_guidance_gate_floor,
                    front_memory_high_threshold=self.front_memory_high_threshold,
                    front_memory_quality_threshold=self.front_memory_quality_threshold,
                    front_memory_noise_threshold=self.front_memory_noise_threshold,
                    front_memory_min_alignment=self.front_memory_min_alignment,
                    front_memory_quality_power=self.front_memory_quality_power,
                    front_memory_residual_ratio_max=self.front_memory_residual_ratio_max,
                    front_memory_query_mask_power=self.front_memory_query_mask_power,
                    front_memory_query_mask_floor=self.front_memory_query_mask_floor,
                    front_memory_train_gate_gradient_multiplier=(
                        self.front_memory_train_gate_gradient_multiplier
                    ),
                    front_memory_max_area_std=self.front_memory_max_area_std,
                    front_memory_min_area_std=self.front_memory_min_area_std,
                    front_memory_min_valid_frame_ratio=(
                        self.front_memory_min_valid_frame_ratio
                    ),
                    front_memory_max_area_jump=self.front_memory_max_area_jump,
                    front_memory_max_spatial_noise=(
                        self.front_memory_max_spatial_noise
                    ),
                    front_memory_min_high_active_ratio=(
                        self.front_memory_min_high_active_ratio
                    ),
                ).to(device, dtype=weight_dtype)
                for _ in range(self.num_cross_attn)
            ]
        )

    def set_front_guidance(
        self,
        enable: bool,
        target_scale: Optional[float] = None,
        boost_strength: Optional[float] = None,
        mode: Optional[str] = None,
        min_area: Optional[float] = None,
        max_area: Optional[float] = None,
        dropout_prob: Optional[float] = None,
        learnable_front_guidance: Optional[bool] = None,
        alpha_max: Optional[float] = None,
        alpha_init: Optional[float] = None,
        adaptive_front_guidance: Optional[bool] = None,
        gate_max: Optional[float] = None,
        gate_init: Optional[float] = None,
        gate_floor: Optional[float] = None,
        memory_high_threshold: Optional[float] = None,
        memory_quality_threshold: Optional[float] = None,
        memory_noise_threshold: Optional[float] = None,
        memory_min_alignment: Optional[float] = None,
        memory_quality_power: Optional[float] = None,
        memory_residual_ratio_max: Optional[float] = None,
        memory_query_mask_power: Optional[float] = None,
        memory_query_mask_floor: Optional[float] = None,
        memory_train_gate_gradient_multiplier: Optional[float] = None,
        memory_max_area_std: Optional[float] = None,
        memory_min_area_std: Optional[float] = None,
        memory_min_valid_frame_ratio: Optional[float] = None,
        memory_max_area_jump: Optional[float] = None,
        memory_max_spatial_noise: Optional[float] = None,
        memory_min_high_active_ratio: Optional[float] = None,
    ):
        self.enable_front_guidance = bool(enable)
        if target_scale is not None:
            self.front_guidance_target_scale = float(target_scale)
        if boost_strength is not None:
            self.front_guidance_boost_strength = float(boost_strength)
        if mode is not None:
            self.front_guidance_mode = str(mode)
        if min_area is not None:
            self.front_guidance_min_area = float(min_area)
        if max_area is not None:
            self.front_guidance_max_area = float(max_area)
        if dropout_prob is not None:
            self.front_guidance_dropout_prob = float(dropout_prob)
        if learnable_front_guidance is not None:
            self.learnable_front_guidance = bool(learnable_front_guidance)
        if alpha_max is not None:
            self.front_guidance_alpha_max = float(alpha_max)
        if alpha_init is not None:
            self.front_guidance_alpha_init = float(alpha_init)
        if adaptive_front_guidance is not None:
            self.adaptive_front_guidance = bool(adaptive_front_guidance)
        if gate_max is not None:
            self.front_guidance_gate_max = float(gate_max)
        if gate_init is not None:
            self.front_guidance_gate_init = float(gate_init)
        if gate_floor is not None:
            self.front_guidance_gate_floor = float(gate_floor)
        if memory_high_threshold is not None:
            self.front_memory_high_threshold = float(memory_high_threshold)
        if memory_quality_threshold is not None:
            self.front_memory_quality_threshold = float(memory_quality_threshold)
        if memory_noise_threshold is not None:
            self.front_memory_noise_threshold = float(memory_noise_threshold)
        if memory_min_alignment is not None:
            self.front_memory_min_alignment = float(memory_min_alignment)
        if memory_quality_power is not None:
            self.front_memory_quality_power = float(memory_quality_power)
        if memory_residual_ratio_max is not None:
            self.front_memory_residual_ratio_max = float(memory_residual_ratio_max)
        if memory_query_mask_power is not None:
            self.front_memory_query_mask_power = float(memory_query_mask_power)
        if memory_query_mask_floor is not None:
            self.front_memory_query_mask_floor = float(memory_query_mask_floor)
        if memory_train_gate_gradient_multiplier is not None:
            self.front_memory_train_gate_gradient_multiplier = float(
                memory_train_gate_gradient_multiplier
            )
        if memory_max_area_std is not None:
            self.front_memory_max_area_std = float(memory_max_area_std)
        if memory_min_area_std is not None:
            self.front_memory_min_area_std = float(memory_min_area_std)
        if memory_min_valid_frame_ratio is not None:
            self.front_memory_min_valid_frame_ratio = float(
                memory_min_valid_frame_ratio
            )
        if memory_max_area_jump is not None:
            self.front_memory_max_area_jump = float(memory_max_area_jump)
        if memory_max_spatial_noise is not None:
            self.front_memory_max_spatial_noise = float(memory_max_spatial_noise)
        if memory_min_high_active_ratio is not None:
            self.front_memory_min_high_active_ratio = float(
                memory_min_high_active_ratio
            )

        self.register_to_config(
            enable_front_guidance=self.enable_front_guidance,
            front_guidance_target_scale=self.front_guidance_target_scale,
            front_guidance_boost_strength=self.front_guidance_boost_strength,
            front_guidance_mode=self.front_guidance_mode,
            front_guidance_min_area=self.front_guidance_min_area,
            front_guidance_max_area=self.front_guidance_max_area,
            front_guidance_dropout_prob=self.front_guidance_dropout_prob,
            learnable_front_guidance=self.learnable_front_guidance,
            front_guidance_alpha_max=self.front_guidance_alpha_max,
            front_guidance_alpha_init=self.front_guidance_alpha_init,
            adaptive_front_guidance=self.adaptive_front_guidance,
            front_guidance_gate_max=self.front_guidance_gate_max,
            front_guidance_gate_init=self.front_guidance_gate_init,
            front_guidance_gate_floor=self.front_guidance_gate_floor,
            front_memory_high_threshold=self.front_memory_high_threshold,
            front_memory_quality_threshold=self.front_memory_quality_threshold,
            front_memory_noise_threshold=self.front_memory_noise_threshold,
            front_memory_min_alignment=self.front_memory_min_alignment,
            front_memory_quality_power=self.front_memory_quality_power,
            front_memory_residual_ratio_max=self.front_memory_residual_ratio_max,
            front_memory_query_mask_power=self.front_memory_query_mask_power,
            front_memory_query_mask_floor=self.front_memory_query_mask_floor,
            front_memory_train_gate_gradient_multiplier=(
                self.front_memory_train_gate_gradient_multiplier
            ),
            front_memory_max_area_std=self.front_memory_max_area_std,
            front_memory_min_area_std=self.front_memory_min_area_std,
            front_memory_min_valid_frame_ratio=(
                self.front_memory_min_valid_frame_ratio
            ),
            front_memory_max_area_jump=self.front_memory_max_area_jump,
            front_memory_max_spatial_noise=self.front_memory_max_spatial_noise,
            front_memory_min_high_active_ratio=(
                self.front_memory_min_high_active_ratio
            ),
        )

        if hasattr(self, "perceiver_cross_attention"):
            for attn in self.perceiver_cross_attention:
                attn.enable_front_guidance = self.enable_front_guidance
                attn.target_guidance_scale = self.front_guidance_target_scale
                attn.boost_strength = self.front_guidance_boost_strength
                attn.front_guidance_mode = self.front_guidance_mode
                attn.front_guidance_min_area = self.front_guidance_min_area
                attn.front_guidance_max_area = self.front_guidance_max_area
                attn.front_guidance_dropout_prob = self.front_guidance_dropout_prob
                attn.learnable_front_guidance = self.learnable_front_guidance
                attn.front_guidance_alpha_max = self.front_guidance_alpha_max
                attn.front_guidance_alpha_init = self.front_guidance_alpha_init
                attn.adaptive_front_guidance = self.adaptive_front_guidance
                attn.front_guidance_gate_max = self.front_guidance_gate_max
                attn.front_guidance_gate_init = self.front_guidance_gate_init
                attn.front_guidance_gate_floor = self.front_guidance_gate_floor
                attn.front_memory_high_threshold = self.front_memory_high_threshold
                attn.front_memory_quality_threshold = self.front_memory_quality_threshold
                attn.front_memory_noise_threshold = self.front_memory_noise_threshold
                attn.front_memory_min_alignment = self.front_memory_min_alignment
                attn.front_memory_quality_power = self.front_memory_quality_power
                attn.front_memory_residual_ratio_max = self.front_memory_residual_ratio_max
                attn.front_memory_query_mask_power = self.front_memory_query_mask_power
                attn.front_memory_query_mask_floor = self.front_memory_query_mask_floor
                attn.front_memory_train_gate_gradient_multiplier = (
                    self.front_memory_train_gate_gradient_multiplier
                )
                attn.front_memory_max_area_std = self.front_memory_max_area_std
                attn.front_memory_min_area_std = self.front_memory_min_area_std
                attn.front_memory_min_valid_frame_ratio = (
                    self.front_memory_min_valid_frame_ratio
                )
                attn.front_memory_max_area_jump = self.front_memory_max_area_jump
                attn.front_memory_max_spatial_noise = (
                    self.front_memory_max_spatial_noise
                )
                attn.front_memory_min_high_active_ratio = (
                    self.front_memory_min_high_active_ratio
                )
                if self.learnable_front_guidance:
                    attn.init_front_guidance_alpha()
                elif hasattr(attn, "front_guidance_alpha"):
                    attn.front_guidance_alpha.requires_grad_(False)
                if self.adaptive_front_guidance:
                    attn.init_adaptive_front_guidance()
                elif hasattr(attn, "front_guidance_gate_bias"):
                    for name in (
                        "front_guidance_gate_bias",
                        "front_guidance_area_weight",
                        "front_guidance_progress_weight",
                    ):
                        getattr(attn, name).requires_grad_(False)
                if self.front_guidance_mode == "foreground_memory":
                    attn.init_front_memory()
                elif hasattr(attn, "front_memory_out"):
                    attn.front_memory_out.requires_grad_(False)
                    if hasattr(attn, "front_memory_gate_bias"):
                        attn.front_memory_gate_bias.requires_grad_(False)

    def set_front_condition(
        self,
        enable: bool,
        use_boundary: Optional[bool] = None,
        strength_init: Optional[float] = None,
        gate_init: Optional[float] = None,
        gate_max: Optional[float] = None,
        video_scale_init: Optional[float] = None,
        front_scale_init: Optional[float] = None,
        boundary_scale_init: Optional[float] = None,
    ):
        self.enable_front_condition = bool(enable)
        if use_boundary is not None:
            self.front_condition_use_boundary = bool(use_boundary)
        if strength_init is not None:
            self.front_condition_strength_init = float(strength_init)
        if gate_init is not None:
            self.front_condition_gate_init = float(gate_init)
        if gate_max is not None:
            self.front_condition_gate_max = float(gate_max)
        if video_scale_init is not None:
            self.front_condition_video_scale_init = float(video_scale_init)
        if front_scale_init is not None:
            self.front_condition_front_scale_init = float(front_scale_init)
        if boundary_scale_init is not None:
            self.front_condition_boundary_scale_init = float(boundary_scale_init)

        self.register_to_config(
            enable_front_condition=self.enable_front_condition,
            front_condition_use_boundary=self.front_condition_use_boundary,
            front_condition_strength_init=self.front_condition_strength_init,
            front_condition_gate_init=self.front_condition_gate_init,
            front_condition_gate_max=self.front_condition_gate_max,
            front_condition_video_scale_init=self.front_condition_video_scale_init,
            front_condition_front_scale_init=self.front_condition_front_scale_init,
            front_condition_boundary_scale_init=self.front_condition_boundary_scale_init,
        )

        if self.enable_front_condition:
            self._init_front_condition_adapter()
        elif hasattr(self, "front_condition_adapter"):
            self.front_condition_adapter.strength.requires_grad_(False)
            self.front_condition_adapter.gate_bias.requires_grad_(False)
            self.front_condition_adapter.video_scale.requires_grad_(False)
            self.front_condition_adapter.front_scale.requires_grad_(False)
            self.front_condition_adapter.boundary_scale.requires_grad_(False)
            for param in self.front_condition_adapter.delta_proj.parameters():
                param.requires_grad_(False)

    def get_front_condition_log_values(self):
        if not hasattr(self, "front_condition_adapter"):
            return {}
        adapter = self.front_condition_adapter

        def _tensor_mean(value, default: float = 0.0) -> float:
            if not torch.is_tensor(value):
                return float(value)
            value = value.detach().float()
            if value.numel() == 0:
                return default
            return float(value.mean().cpu())

        def _tensor_norm(value, default: float = 0.0) -> float:
            if not torch.is_tensor(value):
                return float(abs(value))
            value = value.detach().float()
            if value.numel() == 0:
                return default
            return float(value.norm().cpu())

        weight = adapter.delta_proj.weight
        gate_limit = max(0.0, float(adapter.gate_max))
        gate_init = gate_limit * torch.sigmoid(adapter.gate_bias.detach().float())
        logs = {
            "front_condition/strength": _tensor_mean(adapter.strength),
            "front_condition/delta_weight_norm": _tensor_norm(weight),
            "front_condition/gate_limit": gate_limit,
            "front_condition/gate_bias": _tensor_mean(adapter.gate_bias),
            "front_condition/video_scale": _tensor_mean(adapter.video_scale),
            "front_condition/front_scale": _tensor_mean(adapter.front_scale),
            "front_condition/boundary_scale": _tensor_mean(adapter.boundary_scale),
            "front_condition/gate_mean": _tensor_mean(gate_init),
            "front_condition/gate_max": _tensor_mean(gate_init),
            "front_condition/applied_norm": 0.0,
            "front_condition/applied_to_video_norm_ratio": 0.0,
        }
        last_log_values = getattr(adapter, "_last_log_values", {})
        for key, value in last_log_values.items():
            logs[f"front_condition/{key}"] = _tensor_mean(value)
        return logs

    def get_front_guidance_alpha_values(self):
        if not hasattr(self, "perceiver_cross_attention"):
            return []
        values = []
        for attn in self.perceiver_cross_attention:
            if getattr(attn, "learnable_front_guidance", False):
                if not hasattr(attn, "front_guidance_alpha"):
                    values.append(0.0)
                    continue
                raw_alpha = attn._front_guidance_raw_alpha().detach().float()
                if raw_alpha.numel() == 0:
                    values.append(0.0)
                    continue
                alpha = torch.clamp(
                    raw_alpha,
                    min=0.0,
                    max=max(0.0, float(attn.front_guidance_alpha_max)),
                )
                values.append(float(alpha.cpu()))
            else:
                values.append(float(attn.target_guidance_scale))
        return values

    def get_front_guidance_raw_alpha_values(self):
        if not hasattr(self, "perceiver_cross_attention"):
            return []
        values = []
        for attn in self.perceiver_cross_attention:
            if getattr(attn, "learnable_front_guidance", False):
                if not hasattr(attn, "front_guidance_alpha"):
                    values.append(0.0)
                    continue
                raw_alpha = attn._front_guidance_raw_alpha().detach().float()
                if raw_alpha.numel() == 0:
                    values.append(0.0)
                    continue
                values.append(float(raw_alpha.cpu()))
            else:
                values.append(float(attn.target_guidance_scale))
        return values

    def get_front_guidance_alpha_delta_values(self):
        if not hasattr(self, "perceiver_cross_attention"):
            return []
        values = []
        for attn in self.perceiver_cross_attention:
            if getattr(attn, "learnable_front_guidance", False):
                if not hasattr(attn, "front_guidance_alpha"):
                    values.append(0.0)
                    continue
                alpha_delta = attn.front_guidance_alpha.detach().float()
                if alpha_delta.numel() == 0:
                    values.append(0.0)
                    continue
                values.append(float(alpha_delta.cpu()))
            else:
                values.append(0.0)
        return values

    def get_front_guidance_gate_values(self):
        if not hasattr(self, "perceiver_cross_attention"):
            return []
        values = []
        for attn in self.perceiver_cross_attention:
            if (
                getattr(attn, "front_guidance_mode", "") == "foreground_memory"
                and hasattr(attn, "front_memory_gate_bias")
            ):
                gate_max = max(0.0, float(attn.front_guidance_gate_max))
                gate_floor = max(
                    0.0,
                    min(
                        gate_max,
                        float(getattr(attn, "front_guidance_gate_floor", 0.0)),
                    ),
                )
                gate_span = max(0.0, gate_max - gate_floor)
                gate = gate_floor + gate_span * torch.sigmoid(
                    attn.front_memory_gate_bias.detach().float()
                )
                if gate.numel() == 0:
                    values.append(0.0)
                else:
                    values.append(float(gate.mean().cpu()))
                continue
            if not getattr(attn, "adaptive_front_guidance", False):
                values.append(0.0)
                continue
            if not hasattr(attn, "front_guidance_gate_bias"):
                values.append(0.0)
                continue
            raw_gate = (
                torch.as_tensor(
                    getattr(attn, "front_guidance_gate_init", 0.0),
                    dtype=torch.float32,
                    device=attn.front_guidance_gate_bias.device,
                )
                + attn.front_guidance_gate_bias.detach().float()
            )
            gate_max = max(0.0, float(attn.front_guidance_gate_max))
            gate_floor = max(
                0.0,
                min(gate_max, float(getattr(attn, "front_guidance_gate_floor", 0.0))),
            )
            gate = torch.clamp(
                raw_gate,
                min=gate_floor,
                max=gate_max,
            )
            gate = gate.detach().float()
            if gate.numel() == 0:
                values.append(0.0)
            else:
                values.append(float(gate.mean().cpu()))
        return values

    def get_front_guidance_log_values(self):
        if not hasattr(self, "perceiver_cross_attention"):
            return {}

        def _mean_or_none(values):
            tensors = []
            for value in values:
                if torch.is_tensor(value):
                    value = value.detach().float()
                    if value.numel() > 0:
                        tensors.append(value.mean())
            if not tensors:
                return None
            return float(torch.stack(tensors).mean().cpu())

        gates = []
        areas = []
        valid_areas = []
        memory_qualities = []
        memory_noises = []
        memory_alignments = []
        memory_core_areas = []
        memory_boundary_areas = []
        memory_enabled = []
        memory_sample_gates = []
        memory_area_stds = []
        memory_area_jumps = []
        memory_valid_frame_ratios = []
        memory_high_active_ratios = []
        memory_residual_ratios = []
        memory_query_gates = []
        memory_train_gate_gradient_multipliers = []
        gate_biases = []
        area_weights = []
        progress_weights = []
        for attn in self.perceiver_cross_attention:
            if (
                getattr(attn, "adaptive_front_guidance", False)
                or getattr(attn, "front_guidance_mode", "") == "foreground_memory"
            ):
                gates.append(getattr(attn, "_last_front_guidance_gate", None))
                areas.append(getattr(attn, "_last_front_guidance_area", None))
                valid_areas.append(getattr(attn, "_last_front_guidance_valid_area", None))
            if getattr(attn, "front_guidance_mode", "") == "foreground_memory":
                memory_qualities.append(getattr(attn, "_last_front_memory_quality", None))
                memory_noises.append(getattr(attn, "_last_front_memory_noise", None))
                memory_alignments.append(getattr(attn, "_last_front_memory_alignment", None))
                memory_core_areas.append(getattr(attn, "_last_front_memory_core_area", None))
                memory_boundary_areas.append(getattr(attn, "_last_front_memory_boundary_area", None))
                memory_enabled.append(getattr(attn, "_last_front_memory_enabled", None))
                memory_sample_gates.append(
                    getattr(attn, "_last_front_memory_sample_gate", None)
                )
                memory_area_stds.append(
                    getattr(attn, "_last_front_memory_area_std", None)
                )
                memory_area_jumps.append(
                    getattr(attn, "_last_front_memory_max_area_jump", None)
                )
                memory_valid_frame_ratios.append(
                    getattr(attn, "_last_front_memory_valid_frame_ratio", None)
                )
                memory_high_active_ratios.append(
                    getattr(attn, "_last_front_memory_high_active_ratio", None)
                )
                memory_residual_ratios.append(getattr(attn, "_last_front_memory_residual_ratio", None))
                memory_query_gates.append(getattr(attn, "_last_front_memory_query_gate", None))
                memory_train_gate_gradient_multipliers.append(
                    getattr(
                        attn,
                        "_last_front_memory_train_gate_gradient_multiplier",
                        None,
                    )
                )
                if hasattr(attn, "front_memory_gate_bias"):
                    gate_biases.append(attn.front_memory_gate_bias)
                continue
            if getattr(attn, "adaptive_front_guidance", False) and hasattr(attn, "front_guidance_gate_bias"):
                gate_biases.append(attn.front_guidance_gate_bias)
                area_weights.append(attn.front_guidance_area_weight)
                progress_weights.append(attn.front_guidance_progress_weight)

        logs = {}
        gate_param_values = self.get_front_guidance_gate_values()
        if gate_param_values:
            logs["front_guidance/gate_param_mean"] = (
                sum(gate_param_values) / len(gate_param_values)
            )
            logs["front_guidance/gate_param_max"] = max(gate_param_values)
        for key, values in (
            ("front_guidance/gate_mean", gates),
            ("front_guidance/mask_area_mean", areas),
            ("front_guidance/valid_area_mean", valid_areas),
            ("front_guidance/gate_bias", gate_biases),
            ("front_guidance/area_weight", area_weights),
            ("front_guidance/progress_weight", progress_weights),
            ("front_memory/quality_mean", memory_qualities),
            ("front_memory/noise_mean", memory_noises),
            ("front_memory/alignment_mean", memory_alignments),
            ("front_memory/core_area_mean", memory_core_areas),
            ("front_memory/boundary_area_mean", memory_boundary_areas),
            ("front_memory/enabled_ratio", memory_enabled),
            ("front_memory/sample_gate_mean", memory_sample_gates),
            ("front_memory/area_std_mean", memory_area_stds),
            ("front_memory/max_area_jump_mean", memory_area_jumps),
            ("front_memory/valid_frame_ratio_mean", memory_valid_frame_ratios),
            ("front_memory/high_active_ratio_mean", memory_high_active_ratios),
            ("front_memory/residual_norm_ratio", memory_residual_ratios),
            ("front_memory/query_gate_mean", memory_query_gates),
            (
                "front_memory/train_gate_gradient_multiplier",
                memory_train_gate_gradient_multipliers,
            ),
        ):
            mean_value = _mean_or_none(values)
            if mean_value is not None:
                logs[key] = mean_value
        return logs

    def front_guidance_gate_regularization(self):
        if not hasattr(self, "perceiver_cross_attention"):
            return None
        values = []
        for attn in self.perceiver_cross_attention:
            value = getattr(attn, "_last_front_guidance_gate_for_loss", None)
            if torch.is_tensor(value):
                values.append(value.float())
        if not values:
            return None
        return torch.stack(values).mean()

    def enable_ffn_expert_routers(self, rank: int = 64, scale: float = 1.0) -> None:
        self.latent_ffn_expert_rank = int(rank)
        self.latent_ffn_expert_scale = float(scale)
        for block in self.transformer_blocks:
            block.enable_ffn_expert_router(rank=rank, scale=scale)

    def has_ffn_expert_routers(self) -> bool:
        return all(
            getattr(block, "ffn_expert_router", None) is not None
            for block in self.transformer_blocks
        )

    def _validate_ffn_expert_routes(
        self,
        routes: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if routes is None:
            return None
        if not self.has_ffn_expert_routers():
            raise RuntimeError(
                "latent FFN expert routes were provided before expert routers "
                "were initialized."
            )
        if routes.dim() != 3 or routes.size(-1) != 3:
            raise ValueError(
                "latent_ffn_expert_routes must be [B, N, 3], got "
                f"{tuple(routes.shape)}"
            )
        if routes.size(0) != hidden_states.size(0) or routes.size(1) != hidden_states.size(1):
            raise ValueError(
                "latent_ffn_expert_routes must match video token shape, got "
                f"{tuple(routes.shape[:2])} vs {tuple(hidden_states.shape[:2])}"
            )
        return routes.to(device=hidden_states.device)

    def _prepare_front_token_mask(
        self,
        front_latents: torch.Tensor,
        cross_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Expected front input either [B, C, F, H, W] or [B, F, C, H, W].
        if front_latents.dim() != 5:
            raise ValueError(
                f"front_latents must be 5D, got shape {tuple(front_latents.shape)}"
            )

        if front_latents.shape[1] in (1, 3):
            front = front_latents
        elif front_latents.shape[2] in (1, 3):
            front = front_latents.permute(0, 2, 1, 3, 4)
        else:
            raise ValueError(
                "Unable to infer channel dimension for front_latents. "
                f"Expected shape [B,C,F,H,W] or [B,F,C,H,W], got {tuple(front_latents.shape)}"
            )

        if front.shape[1] > 1:
            front = front.mean(dim=1, keepdim=True)
        elif front.shape[1] != 1:
            front = front[:, :1]

        patch = int(self.ref_patch_embed.patch_size)
        front = F.avg_pool3d(
            front,
            kernel_size=(1, patch, patch),
            stride=(1, patch, patch),
        )
        # [B, 1, F, H, W] -> [B, F, H*W, 1] -> [B, F*H*W]
        front = front.permute(0, 2, 1, 3, 4).flatten(3).transpose(2, 3).flatten(1, 2)
        front = front.squeeze(-1)

        target_len = cross_hidden_states.shape[1]
        cur_len = front.shape[1]
        if cur_len < target_len:
            front = F.pad(front, (0, target_len - cur_len))
        elif cur_len > target_len:
            front = front[:, :target_len]

        return front

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedCogVideoXAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError(
                    "`fuse_qkv_projections()` is not supported for models having added KV projections."
                )

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedCogVideoXAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def forward(
        self,
        hidden_states: torch.Tensor,  # noise
        encoder_hidden_states: torch.Tensor,  # text
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        inpaint_latents: Optional[torch.Tensor] = None,  # condition
        cross_latents: Optional[torch.Tensor] = None,
        front_latents: Optional[torch.Tensor] = None,
        train_progress: float = 1.0,
        latent_ffn_expert_routes: Optional[torch.Tensor] = None,
        latent_ffn_expert_scale: Optional[Union[float, torch.Tensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_dict: bool = True,
    ):
        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        # [2, 13, 16, 48, 84] cat [2, 13, 17, 48, 84] = [2, 13, 33, 48, 84]
        hidden_states = torch.concat([hidden_states, inpaint_latents], 2)
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        text_seq_length = encoder_hidden_states.shape[1]
        if (
            self.enable_front_condition
            and front_latents is not None
            and hasattr(self, "front_condition_adapter")
        ):
            video_tokens = hidden_states[:, text_seq_length:]
            front_condition_applied = self.front_condition_adapter(
                front_latents,
                video_tokens,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            hidden_states = hidden_states.clone()
            hidden_states[:, text_seq_length:] = (
                hidden_states[:, text_seq_length:] + front_condition_applied
            )
        front_token_mask = None
        ref_token_shape = None
        if self.is_train_cross:
            patch = int(self.ref_patch_embed.patch_size)
            ref_token_shape = (
                int(cross_latents.shape[1]),
                int(cross_latents.shape[-2] // patch),
                int(cross_latents.shape[-1] // patch),
            )
            cross_hidden_states = self.ref_patch_embed(cross_latents)
            if self.enable_front_guidance and front_latents is not None:
                front_token_mask = self._prepare_front_token_mask(
                    front_latents, cross_hidden_states
                )

        # 3. Position embedding
        if not self.config.use_rotary_positional_embeddings:
            seq_length = height * width * num_frames // (self.config.patch_size**2)
            # pos_embeds = self.pos_embedding[:, : text_seq_length + seq_length]
            pos_embeds = self.pos_embedding
            emb_size = hidden_states.size()[-1]
            pos_embeds_without_text = pos_embeds[:, text_seq_length:].view(
                1,
                self.post_time_compression_frames,
                self.post_patch_height,
                self.post_patch_width,
                emb_size,
            )
            pos_embeds_without_text = pos_embeds_without_text.permute([0, 4, 1, 2, 3])
            pos_embeds_without_text = F.interpolate(
                pos_embeds_without_text,
                size=[
                    self.post_time_compression_frames,
                    height // self.config.patch_size,
                    width // self.config.patch_size,
                ],
                mode='trilinear',
                align_corners=False,
            )
            pos_embeds_without_text = pos_embeds_without_text.permute(
                [0, 2, 3, 4, 1]
            ).view(1, -1, emb_size)
            pos_embeds = torch.cat(
                [pos_embeds[:, :text_seq_length], pos_embeds_without_text], dim=1
            )
            pos_embeds = pos_embeds[:, : text_seq_length + seq_length]
            hidden_states = hidden_states + pos_embeds
            hidden_states = self.embedding_dropout(hidden_states)
        # seperate
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]
        latent_ffn_expert_routes = self._validate_ffn_expert_routes(
            latent_ffn_expert_routes,
            hidden_states,
        )

        # 4. Transformer blocks

        ca_idx = 0
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states, encoder_hidden_states = (
                    torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        emb,
                        image_rotary_emb,
                        latent_ffn_expert_routes,
                        latent_ffn_expert_scale,
                        **ckpt_kwargs,
                    )
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    ffn_expert_routes=latent_ffn_expert_routes,
                    ffn_expert_scale=latent_ffn_expert_scale,
                )
            if self.is_train_cross:
                if i % self.cross_attn_interval == 0:
                    hidden_states = hidden_states + self.perceiver_cross_attention[
                        ca_idx
                    ](
                        cross_hidden_states,
                        hidden_states,
                        ref_mask=front_token_mask,
                        train_progress=train_progress,
                        ref_token_shape=ref_token_shape,
                    )  # torch.Size([2, 32, 2048])  torch.Size([2, 17550, 3072])
                    ca_idx += 1

        # if not self.config.use_rotary_positional_embeddings:
        #     # CogVideoX-2B
        #     hidden_states = self.norm_final(hidden_states)
        # else:
        # use CogVideoX-5B
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        hidden_states = self.norm_final(hidden_states)
        hidden_states = hidden_states[:, text_seq_length:]

        # 5. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 6. Unpatchify
        p = self.config.patch_size
        output = hidden_states.reshape(
            batch_size, num_frames, height // p, width // p, channels, p, p
        )
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    @classmethod
    def _load_transformer_state_dict(cls, pretrained_model_path: str):
        from diffusers.utils import WEIGHTS_NAME

        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")
        if os.path.exists(model_file):
            payload = torch.load(model_file, map_location="cpu")
            return payload["state_dict"] if "state_dict" in payload else payload
        if os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file

            return load_file(model_file_safetensors)

        safetensor_files = sorted(
            path
            for path in glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
            if not path.endswith(".index.safetensors")
        )
        if safetensor_files:
            from safetensors.torch import load_file

            state_dict = {}
            for safetensor_file in safetensor_files:
                state_dict.update(load_file(safetensor_file))
            return state_dict

        for filename in ("diffusion_pytorch_model.bin", "pytorch_model.bin"):
            candidate = os.path.join(pretrained_model_path, filename)
            if os.path.isfile(candidate):
                payload = torch.load(candidate, map_location="cpu")
                return payload["state_dict"] if "state_dict" in payload else payload

        raise FileNotFoundError(
            "No transformer weights found under "
            f"{pretrained_model_path}. Expected *.safetensors or *.bin weights."
        )

    @staticmethod
    def _infer_ffn_expert_rank(state_dict) -> Optional[int]:
        rank_shapes = {
            tuple(weight.shape)
            for key, weight in state_dict.items()
            if "ffn_expert_router" in key and key.endswith("experts.0.0.weight")
        }
        if not rank_shapes:
            return None
        if len(rank_shapes) != 1:
            raise ValueError(
                "Inconsistent FFN expert rank shapes in checkpoint: "
                f"{sorted(rank_shapes)}"
            )
        shape = next(iter(rank_shapes))
        if len(shape) != 2:
            raise ValueError(f"Expected 2D expert rank weight, got shape {shape}")
        return int(shape[0])

    @classmethod
    def from_pretrained_expert_aware(
        cls,
        pretrained_model_path,
        subfolder=None,
        ffn_expert_scale: float = 1.0,
        transformer_additional_kwargs=None,
    ):
        if transformer_additional_kwargs is None:
            transformer_additional_kwargs = {}
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)

        config_file = os.path.join(pretrained_model_path, "config.json")
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        print(
            "Loading expert-aware 3D transformer weights from "
            f"{pretrained_model_path} ..."
        )
        state_dict = cls._load_transformer_state_dict(pretrained_model_path)
        model = cls.from_config(config, **transformer_additional_kwargs)

        expert_keys = [key for key in state_dict if "ffn_expert_router" in key]
        if expert_keys:
            rank = cls._infer_ffn_expert_rank(state_dict)
            if rank is None:
                raise ValueError(
                    "Found FFN expert router weights but could not infer rank from "
                    "experts.0.0.weight."
                )
            model.enable_ffn_expert_routers(rank=rank, scale=ffn_expert_scale)
            print(
                f"Enabled FFN expert routers before loading: rank={rank}, "
                f"scale={ffn_expert_scale}, keys={len(expert_keys)}"
            )

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Transformer checkpoint mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return model

    @classmethod
    def from_pretrained_2d(
        cls, pretrained_model_path, subfolder=None, transformer_additional_kwargs={}
    ):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(
            f"loaded 3D transformer's pretrained weights from {pretrained_model_path} ..."
        )

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        from diffusers.utils import WEIGHTS_NAME

        model = cls.from_config(config, **transformer_additional_kwargs)
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file, safe_open

            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file, safe_open

            model_files_safetensors = glob.glob(
                os.path.join(pretrained_model_path, "*.safetensors")
            )
            state_dict = {}
            for model_file_safetensors in model_files_safetensors:
                _state_dict = load_file(model_file_safetensors)
                for key in _state_dict:
                    state_dict[key] = _state_dict[key]

        if (
            model.state_dict()['patch_embed.proj.weight'].size()
            != state_dict['patch_embed.proj.weight'].size()
        ):
            new_shape = model.state_dict()['patch_embed.proj.weight'].size()
            if len(new_shape) == 5:
                state_dict['patch_embed.proj.weight'] = (
                    state_dict['patch_embed.proj.weight']
                    .unsqueeze(2)
                    .expand(new_shape)
                    .clone()
                )
                state_dict['patch_embed.proj.weight'][:, :, :-1] = 0
            else:
                if (
                    model.state_dict()['patch_embed.proj.weight'].size()[1]
                    > state_dict['patch_embed.proj.weight'].size()[1]
                ):
                    model.state_dict()['patch_embed.proj.weight'][
                        :, : state_dict['patch_embed.proj.weight'].size()[1], :, :
                    ] = state_dict['patch_embed.proj.weight']
                    model.state_dict()['patch_embed.proj.weight'][
                        :, state_dict['patch_embed.proj.weight'].size()[1] :, :, :
                    ] = 0
                    state_dict['patch_embed.proj.weight'] = model.state_dict()[
                        'patch_embed.proj.weight'
                    ]
                else:
                    model.state_dict()['patch_embed.proj.weight'][
                        :, :, :, :
                    ] = state_dict['patch_embed.proj.weight'][
                        :,
                        : model.state_dict()['patch_embed.proj.weight'].size()[1],
                        :,
                        :,
                    ]
                    state_dict['patch_embed.proj.weight'] = model.state_dict()[
                        'patch_embed.proj.weight'
                    ]

        tmp_state_dict = {}
        model_state = model.state_dict()
        for key in state_dict:
            # Expert-router tensors are optional when latent experts are disabled.
            if "ffn_expert_router" in key:
                continue
            if key in model_state and model_state[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")
        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        print(m)

        params = [p.numel() if "mamba" in n else 0 for n, p in model.named_parameters()]
        print(f"### Mamba Parameters: {sum(params) / 1e6} M")

        params = [
            p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()
        ]
        print(f"### attn1 Parameters: {sum(params) / 1e6} M")

        return model

    @classmethod
    def from_pretrained_cus(
        cls,
        pretrained_model_path,
        subfolder=None,
        config_path=None,
        transformer_additional_kwargs={},
    ):
        if subfolder:
            config_path = config_path or pretrained_model_path
            config_file = os.path.join(config_path, subfolder, 'config.json')
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        else:
            config_file = os.path.join(
                config_path or pretrained_model_path, 'config.json'
            )

        print(
            f"Loading 3D transformer's pretrained weights from {pretrained_model_path} ..."
        )

        # Check if config file exists
        if not os.path.isfile(config_file):
            raise RuntimeError(f"Configuration file '{config_file}' does not exist")

        # Load the configuration
        with open(config_file, "r") as f:
            config = json.load(f)

        from diffusers.utils import WEIGHTS_NAME

        model = cls.from_config(config, **transformer_additional_kwargs)
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file

            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file

            model_files_safetensors = glob.glob(
                os.path.join(pretrained_model_path, "*.safetensors")
            )
            state_dict = {}
            for model_file_safetensors in model_files_safetensors:
                _state_dict = load_file(model_file_safetensors)
                for key in _state_dict:
                    state_dict[key] = _state_dict[key]

        if (
            model.state_dict()['patch_embed.proj.weight'].size()
            != state_dict['patch_embed.proj.weight'].size()
        ):
            new_shape = model.state_dict()['patch_embed.proj.weight'].size()
            if len(new_shape) == 5:
                state_dict['patch_embed.proj.weight'] = (
                    state_dict['patch_embed.proj.weight']
                    .unsqueeze(2)
                    .expand(new_shape)
                    .clone()
                )
                state_dict['patch_embed.proj.weight'][:, :, :-1] = 0
            else:
                if (
                    model.state_dict()['patch_embed.proj.weight'].size()[1]
                    > state_dict['patch_embed.proj.weight'].size()[1]
                ):
                    model.state_dict()['patch_embed.proj.weight'][
                        :, : state_dict['patch_embed.proj.weight'].size()[1], :, :
                    ] = state_dict['patch_embed.proj.weight']
                    model.state_dict()['patch_embed.proj.weight'][
                        :, state_dict['patch_embed.proj.weight'].size()[1] :, :, :
                    ] = 0
                    state_dict['patch_embed.proj.weight'] = model.state_dict()[
                        'patch_embed.proj.weight'
                    ]
                else:
                    model.state_dict()['patch_embed.proj.weight'][
                        :, :, :, :
                    ] = state_dict['patch_embed.proj.weight'][
                        :,
                        : model.state_dict()['patch_embed.proj.weight'].size()[1],
                        :,
                        :,
                    ]
                    state_dict['patch_embed.proj.weight'] = model.state_dict()[
                        'patch_embed.proj.weight'
                    ]

        tmp_state_dict = {}
        for key in state_dict:
            if (
                key in model.state_dict().keys()
                and model.state_dict()[key].size() == state_dict[key].size()
            ):
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")
        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        print(m)

        params = [p.numel() if "mamba" in n else 0 for n, p in model.named_parameters()]
        print(f"### Mamba Parameters: {sum(params) / 1e6} M")

        params = [
            p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()
        ]
        print(f"### attn1 Parameters: {sum(params) / 1e6} M")

        return model


if __name__ == '__main__':
    device = "cuda:0"
    weight_dtype = torch.bfloat16
    model_path = "/group/40075/wangboyu/CogVideoX-Fun/CogVideoX-Fun-V1.1-5b-InP"

    transformer_additional_kwargs = {
        'is_train_cross': True,
        'cross_attn_in_channels': 16,
        'cross_attn_interval': 2,
        'cross_attn_dim_head': 128,
        'cross_attn_num_heads': 16,
    }

    transformer = CrossTransformer3DModel.from_pretrained_2d(
        model_path,
        subfolder="transformer",
        transformer_additional_kwargs=transformer_additional_kwargs,
    )

    transformer.to(device, dtype=weight_dtype)
    for param in transformer.parameters():
        param.requires_grad = False
    transformer.eval()

    b = 1
    dim = 16
    noisy_latents = torch.ones(b, 13, dim, 60, 90).to(device, dtype=weight_dtype)
    inpaint_latents = torch.ones(b, 13, dim + 1, 60, 90).to(device, dtype=weight_dtype)
    # cross_latents          = torch.ones(b, 13, dim, 60, 90).to(device, dtype=weight_dtype)
    cross_latents = torch.ones(b, 1, dim, 60, 90).to(device, dtype=weight_dtype)
    prompt_embeds = torch.ones(b, 226, 4096).to(device, dtype=weight_dtype)
    image_rotary_emb = (
        torch.ones(17550, 64).to(device, dtype=weight_dtype),
        torch.ones(17550, 64).to(device, dtype=weight_dtype),
    )
    timesteps = torch.tensor([311]).to(device, dtype=weight_dtype)
    assert len(timesteps) == b

    model_output = transformer(
        hidden_states=noisy_latents,
        encoder_hidden_states=prompt_embeds,
        timestep=timesteps,
        inpaint_latents=inpaint_latents,
        cross_latents=cross_latents,
        image_rotary_emb=image_rotary_emb,
        return_dict=False,
    )[0]

    print(model_output)
