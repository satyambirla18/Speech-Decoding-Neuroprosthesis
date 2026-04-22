from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device=None, dtype=None):
        device = device or self.inv_freq.device
        dtype = dtype or self.inv_freq.dtype
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, rope_emb):
    return (x * rope_emb.cos()) + (rotate_half(x) * rope_emb.sin())


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class GLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * torch.sigmoid(gate)

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in, padding=padding)

    def forward(self, x):
        return self.conv(x)

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return pad

class Scale(nn.Module):
    def __init__(self, scale: float, fn: nn.Module):
        super().__init__()
        self.scale = scale
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        use_rope: bool = False,
        use_sdpa: bool = True,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        self.attn_dropout = nn.Dropout(dropout)

        self.use_rope = use_rope
        self.use_sdpa = use_sdpa and hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, _ = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        if self.use_rope:
            q, k = apply_rope(q, k)

        if self.use_sdpa:
            attn_mask = mask[:, None, None, :] if mask is not None else None
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False,
            )
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

            if mask is not None:
                key_mask = mask[:, None, None, :]
                dots = dots.masked_fill(~key_mask, float("-inf"))

            attn = dots.softmax(dim=-1)
            attn = self.attn_dropout(attn)
            out = torch.matmul(attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class ConvolutionModule(nn.Module):
    def __init__(self, dim: int, causal: bool = False, expansion_factor: int = 2, kernel_size: int = 31, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            rearrange_layer("b n c -> b c n"),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(inner_dim),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            rearrange_layer("b c n -> b n c"),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class rearrange_layer(nn.Module):
    def __init__(self, pattern: str):
        super().__init__()
        self.pattern = pattern

    def forward(self, x):
        return rearrange(x, self.pattern)

class ConformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ff_mult: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        use_rope: bool = True,
        use_sdpa: bool = True,
        use_conv: bool = True,
        layer_drop: float = 0.0,
    ):
        super().__init__()

        self.ff1 = FeedForward(dim, mult=ff_mult, dropout=dropout)
        self.attn = MultiHeadAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            use_rope=use_rope,
            use_sdpa=use_sdpa,
        )
        self.conv = ConvolutionModule(dim, kernel_size=conv_kernel_size) if use_conv else nn.Identity()
        self.ff2 = FeedForward(dim, mult=ff_mult, dropout=dropout)

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.use_conv = use_conv
        self.layer_drop = float(layer_drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.training and self.layer_drop > 0.0:
            if torch.rand((), device=x.device) < self.layer_drop:
                return x

        x = x + 0.5 * self.dropout(self.ff1(self.norm(x)))
        x = x + self.dropout(self.attn(self.norm(x), mask=mask))
        x = x + self.dropout(self.conv(self.norm(x)))
        x = x + 0.5 * self.dropout(self.ff2(self.norm(x)))
        return x


class Conformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        ff_mult: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        use_rope: bool = True,
        use_sdpa: bool = True,
        use_conv: bool = True,
        layer_drop: float = 0.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                    use_rope=use_rope,
                    use_sdpa=use_sdpa,
                    use_conv=use_conv,
                    layer_drop=layer_drop,
                )
                for _ in range(depth)
            ]
        )
        self.gradient_checkpointing = bool(gradient_checkpointing)

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and mask is not None:
                def _f(_x, _mask):
                    return layer(_x, mask=_mask)

                try:
                    x = checkpoint(_f, x, mask, use_reentrant=False)
                except TypeError:
                    x = checkpoint(_f, x, mask)
            else:
                x = layer(x, mask=mask)
        return x


class ConformerCTC(nn.Module):

    def __init__(
        self,
        input_dim: int = 512,
        model_dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 32,
        vocab_size: int = 42,
        ff_mult: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        use_rope: bool = True,
        use_sdpa: bool = True,
        use_conv: bool = True,
        layer_drop: float = 0.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.Dropout(dropout),
        )
        self.encoder = Conformer(
            dim=model_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            ff_mult=ff_mult,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
            use_rope=use_rope,
            use_sdpa=use_sdpa,
            use_conv=use_conv,
            layer_drop=layer_drop,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.classifier = nn.Linear(model_dim, vocab_size)

    def forward(self, x: torch.Tensor, x_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        mask = None
        if x_len is not None:
            B, T, _ = x.shape
            rng = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
            mask = rng < x_len.unsqueeze(1)

        h = self.input_proj(x)
        h = self.encoder(h, mask=mask)
        logits = self.classifier(h)
        return logits