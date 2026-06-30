"""
Non-causal variant of NeuralAudioCodec for ablation comparison.

Architecture is identical to model.py except:
  - Convolutions use symmetric padding (can see future samples)
  - Attention uses full bidirectional context (no causal mask)

NonCausalConv1d wraps nn.Conv1d under a `.conv` attribute so that its
state-dict keys match CausalConv1d exactly — weights from a trained
causal checkpoint can be loaded without remapping.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NonCausalConv1d(nn.Module):
    """Symmetric-padding Conv1d with the same .conv state-dict key as CausalConv1d."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=(kernel_size // 2) * dilation, dilation=dilation,
        )

    def forward(self, x):
        return self.conv(x)


class NonCausalAttention(nn.Module):
    """Full bidirectional multi-head attention — no causal mask."""
    def __init__(self, d_model, n_heads, window_size=None, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv      = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(dropout)
        self.scale    = 1.0 / math.sqrt(self.d_k)

    def forward(self, x):
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.matmul(q, k.transpose(-2, -1)).float() * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().reshape(B, T, self.d_model)
        return self.out_proj(out)


class NonCausalTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, window_size=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = NonCausalAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class NonCausalAudioEncoder(nn.Module):
    def __init__(self, d_model=256, n_layers=4, n_heads=8, window_size=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                NonCausalConv1d(1, 32, kernel_size=7, stride=2),
                nn.GroupNorm(4, 32), nn.GELU(),
            ),
            nn.Sequential(
                NonCausalConv1d(32, 64, kernel_size=7, stride=2),
                nn.GroupNorm(8, 64), nn.GELU(),
            ),
            nn.Sequential(
                NonCausalConv1d(64, 128, kernel_size=7, stride=2),
                nn.GroupNorm(16, 128), nn.GELU(),
            ),
            nn.Sequential(
                NonCausalConv1d(128, d_model, kernel_size=3, stride=1),
                nn.GroupNorm(8, d_model), nn.GELU(),
            ),
        ])
        self.transformer_blocks = nn.ModuleList([
            NonCausalTransformerBlock(d_model, n_heads, window_size=window_size, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.transpose(1, 2)
        for block in self.transformer_blocks:
            x = block(x)
        return self.norm(x)


class NonCausalAudioDecoder(nn.Module):
    def __init__(self, d_model=256, n_layers=4, n_heads=8, window_size=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.transformer_blocks = nn.ModuleList([
            NonCausalTransformerBlock(d_model, n_heads, window_size=window_size, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.deconv_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(d_model, 128, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(16, 128), nn.GELU(),
            ),
            nn.Sequential(
                nn.ConvTranspose1d(128, 64, kernel_size=8, stride=2, padding=3, output_padding=0),
                nn.GroupNorm(8, 64), nn.GELU(),
            ),
            nn.Sequential(
                nn.ConvTranspose1d(64, 32, kernel_size=8, stride=2, padding=3, output_padding=0),
                nn.GroupNorm(4, 32), nn.GELU(),
            ),
            nn.Sequential(
                nn.ConvTranspose1d(32, 1, kernel_size=8, stride=2, padding=3, output_padding=0),
            ),
        ])

    def forward(self, x):
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        x = x.transpose(1, 2)
        for layer in self.deconv_layers:
            x = layer(x)
        return torch.tanh(x)


class NonCausalNeuralAudioCodec(nn.Module):
    """
    Bidirectional (non-streaming) variant of NeuralAudioCodec.

    Removes causal constraints from both convolutions and attention so the
    model can use arbitrary future context. Intended as an ablation baseline:
    "what PESQ/STOI ceiling is reachable when the real-time constraint is lifted?"

    State-dict keys are structurally compatible with NeuralAudioCodec so that
    causal model weights can be transferred via load_state_dict(strict=True).
    """
    def __init__(
        self,
        sample_rate=16000,
        hop_length=160,
        d_model=256,
        n_layers=4,
        n_heads=8,
        window_size=None,
        dropout=0.1,
        bottleneck_dim=None,
        temporal_stride=1,
    ):
        super().__init__()
        self.bottleneck_dim  = bottleneck_dim
        self.temporal_stride = temporal_stride

        self.encoder = NonCausalAudioEncoder(d_model, n_layers, n_heads, window_size, dropout)
        self.decoder = NonCausalAudioDecoder(d_model, n_layers, n_heads, window_size, dropout)

        dim = bottleneck_dim if bottleneck_dim is not None else d_model
        if bottleneck_dim is not None:
            self.encoder_proj = nn.Linear(d_model, bottleneck_dim)
            self.decoder_proj = nn.Linear(bottleneck_dim, d_model)
        else:
            self.encoder_proj = None
            self.decoder_proj = None

        if temporal_stride > 1:
            self.temporal_enc = nn.Conv1d(dim, dim, kernel_size=temporal_stride,
                                          stride=temporal_stride, padding=0)
            self.temporal_dec = nn.ConvTranspose1d(dim, dim, kernel_size=temporal_stride,
                                                   stride=temporal_stride, padding=0)
        else:
            self.temporal_enc = None
            self.temporal_dec = None

    def encode(self, x):
        z = self.encoder(x)
        if self.encoder_proj is not None:
            z = self.encoder_proj(z)
        if self.temporal_enc is not None:
            z = self.temporal_enc(z.transpose(1, 2)).transpose(1, 2)
        return z

    def decode(self, z):
        if self.temporal_dec is not None:
            z = self.temporal_dec(z.transpose(1, 2)).transpose(1, 2)
        if self.decoder_proj is not None:
            z = self.decoder_proj(z)
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))
