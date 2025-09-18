from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    seq_len: int = 1024
    hidden_size: int = 1024
    num_layers: int = 8
    nhead: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    emb_size: int = 512


class CasualMultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.nhead = config.nhead
        self.d_head = config.hidden_size // self.nhead
        self.hidden_size = config.hidden_size

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        q, k, v = self.attn(x).split(self.hidden_size, dim=2)
        q = q.view(batch, seq, self.nhead, self.d_head).transpose(1, 2)
        k = k.view(batch, seq, self.nhead, self.d_head).transpose(1, 2)
        v = v.view(batch, seq, self.nhead, self.d_head).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(*x.size())
        attn_output = self.dropout(attn_output)
        output = self.proj(attn_output)
        output = self.dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_size, config.dim_feedforward),
            nn.GELU(),
            nn.Linear(config.dim_feedforward, config.hidden_size),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.attn = CasualMultiHeadAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.num_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        self.seq_len = config.seq_len

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        x = self.final_layer_norm(x)
        return x
