from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0, use_rope: bool = True):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.use_rope = use_rope
        self.rope = RotaryEmbedding(dim_head) if use_rope else None

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        if self.use_rope and self.rope is not None:
            rope_emb = self.rope(n, device=x.device, dtype=x.dtype)
            rope_emb = rope_emb.unsqueeze(0).unsqueeze(0)
            q = apply_rope(q, rope_emb)
            k = apply_rope(k, rope_emb)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            m = mask[:, None, None, :].to(torch.bool)
            dots = dots.masked_fill(~m, float("-inf"))

        attn = dots.softmax(dim=-1)
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
    def __init__(self, dim: int, *, dim_head: int = 64, heads: int = 8, ff_mult: int = 4, conv_expansion_factor: int = 2,
                 conv_kernel_size: int = 31, attn_dropout: float = 0.0, ff_dropout: float = 0.0, conv_dropout: float = 0.0,
                 use_rope: bool = True):
        super().__init__()
        self.ff1 = Scale(0.5, PreNorm(dim, FeedForward(dim, mult=ff_mult, dropout=ff_dropout)))
        self.attn = PreNorm(dim, MultiHeadAttention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout, use_rope=use_rope))
        self.conv = ConvolutionModule(dim, expansion_factor=conv_expansion_factor, kernel_size=conv_kernel_size, dropout=conv_dropout)
        self.ff2 = Scale(0.5, PreNorm(dim, FeedForward(dim, mult=ff_mult, dropout=ff_dropout)))
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        x = x + self.ff1(x)
        x = x + self.attn(x, mask=mask)
        x = x + self.conv(x)
        x = x + self.ff2(x)
        return self.post_norm(x)

class Conformer(nn.Module):
    def __init__(self, dim: int, depth: int, *, dim_head: int = 64, heads: int = 8, ff_mult: int = 4,
                 conv_expansion_factor: int = 2, conv_kernel_size: int = 31, attn_dropout: float = 0.0,
                 ff_dropout: float = 0.0, conv_dropout: float = 0.0, use_rope: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            ConformerBlock(
                dim,
                dim_head=dim_head,
                heads=heads,
                ff_mult=ff_mult,
                conv_expansion_factor=conv_expansion_factor,
                conv_kernel_size=conv_kernel_size,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                conv_dropout=conv_dropout,
                use_rope=use_rope,
            )
            for _ in range(depth)
        ])

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        for layer in self.layers:
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
        dropout: float = 0.1,
        use_rope: bool = True,
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
            ff_dropout=dropout,
            attn_dropout=dropout,
            conv_dropout=dropout,
            use_rope=use_rope,
        )
        self.classifier = nn.Linear(model_dim, vocab_size)

    def forward(self, x: torch.Tensor, x_len: Optional[torch.Tensor] = None):
        mask = None
        if x_len is not None:
            B, T, _ = x.shape
            rng = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
            mask = rng < x_len.unsqueeze(1)

        h = self.input_proj(x)
        h = self.encoder(h, mask=mask)
        logits = self.classifier(h)
        return logits
