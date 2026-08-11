"""
Neural Audio Codec: encoder, spatial bottleneck, temporal bottleneck, decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalConv1d(nn.Module):
    """Causal 1D convolution for streaming (no future context)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=self.padding, dilation=dilation
        )

    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class CausalAttention(nn.Module):
    """Sliding-window causal attention for low latency"""
    def __init__(self, d_model, n_heads, window_size=256, dropout=0.1, fixed_window_mask=False):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size
        # See README "Known Limitations": the window mask below was a no-op
        # (torch.triu masked future positions, already covered by causal_mask,
        # instead of past positions beyond the window) for every checkpoint
        # trained before this flag existed. Defaults to False so existing
        # checkpoints keep running inference exactly as they were trained --
        # only checkpoints explicitly trained with the fix set this True.
        self.fixed_window_mask = fixed_window_mask

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # keep in float32 to avoid fp16 overflow
        attn = torch.matmul(q, k.transpose(-2, -1)).float() * self.scale

        device = x.device
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

        if self.window_size is not None and seq_len > self.window_size:
            if self.fixed_window_mask:
                # Correct: mask positions more than window_size steps in the past.
                window_mask = torch.tril(torch.ones(seq_len, seq_len, device=device),
                                         diagonal=-(self.window_size + 1)).bool()
            else:
                # Original bug, preserved for backward compatibility with
                # checkpoints trained under it (see fixed_window_mask above).
                window_mask = torch.triu(torch.ones(seq_len, seq_len, device=device),
                                         diagonal=self.window_size + 1).bool()
            causal_mask = causal_mask | window_mask

        # -1e4 avoids fp16 overflow on masked positions
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e4)
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with causal attention and feed-forward network"""
    def __init__(self, d_model, n_heads, d_ff=None, window_size=256, dropout=0.1, fixed_window_mask=False):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.attention = CausalAttention(d_model, n_heads, window_size, dropout, fixed_window_mask)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class AudioEncoder(nn.Module):
    """Streaming audio encoder: raw audio → latent representations."""
    def __init__(
        self,
        sample_rate=16000,
        hop_length=160,
        d_model=256,
        n_layers=4,
        n_heads=8,
        window_size=256,
        dropout=0.1,
        fixed_window_mask=False
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.d_model = d_model

        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                CausalConv1d(1, 32, kernel_size=7, stride=2),
                nn.GroupNorm(4, 32),
                nn.GELU(),
            ),
            nn.Sequential(
                CausalConv1d(32, 64, kernel_size=7, stride=2),
                nn.GroupNorm(8, 64),
                nn.GELU(),
            ),
            nn.Sequential(
                CausalConv1d(64, 128, kernel_size=7, stride=2),
                nn.GroupNorm(16, 128),
                nn.GELU(),
            ),
            nn.Sequential(
                CausalConv1d(128, d_model, kernel_size=3, stride=1),
                nn.GroupNorm(8, d_model),
                nn.GELU(),
            ),
        ])

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, window_size=window_size, dropout=dropout,
                             fixed_window_mask=fixed_window_mask)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = x.transpose(1, 2)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        return x


class AudioDecoder(nn.Module):
    """Streaming audio decoder: latent representations → waveform."""
    def __init__(
        self,
        sample_rate=16000,
        hop_length=160,
        d_model=256,
        n_layers=4,
        n_heads=8,
        window_size=256,
        dropout=0.1,
        fixed_window_mask=False
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.d_model = d_model

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, window_size=window_size, dropout=dropout,
                             fixed_window_mask=fixed_window_mask)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.deconv_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(d_model, 128, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(16, 128),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.ConvTranspose1d(128, 64, kernel_size=8, stride=2, padding=3, output_padding=0),
                nn.GroupNorm(8, 64),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.ConvTranspose1d(64, 32, kernel_size=8, stride=2, padding=3, output_padding=0),
                nn.GroupNorm(4, 32),
                nn.GELU(),
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

        for deconv_layer in self.deconv_layers:
            x = deconv_layer(x)

        x = torch.tanh(x)
        return x


class NeuralAudioCodec(nn.Module):
    """
    Complete Neural Audio Codec: Encoder + Decoder

    Bitrate control is via two complementary mechanisms:

    1. bottleneck_dim  — reduces spatial dimension (384 → 32)
    2. temporal_stride — reduces frame rate (2000 Hz → 100 Hz with stride=20)

    With bottleneck_dim=32, temporal_stride=20, and 3-bit quantization:
        32 × 3 × 100 Hz = 9,600 bps = 9.6 kbps  (hard architectural cap)

    Temporal compression uses a learnable strided Conv1d (encode) and
    ConvTranspose1d (decode) so the model can learn the best way to
    summarise/reconstruct across the stride window.
    """
    def __init__(
        self,
        sample_rate=16000,
        hop_length=160,
        d_model=256,
        n_layers=4,
        n_heads=8,
        window_size=256,
        dropout=0.1,
        bottleneck_dim=None,
        temporal_stride=1,
        fixed_window_mask=False,
    ):
        super().__init__()

        self.bottleneck_dim = bottleneck_dim
        self.temporal_stride = temporal_stride
        self.fixed_window_mask = fixed_window_mask

        self.encoder = AudioEncoder(
            sample_rate=sample_rate,
            hop_length=hop_length,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            window_size=window_size,
            dropout=dropout,
            fixed_window_mask=fixed_window_mask
        )

        self.decoder = AudioDecoder(
            sample_rate=sample_rate,
            hop_length=hop_length,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            window_size=window_size,
            dropout=dropout,
            fixed_window_mask=fixed_window_mask
        )

        # Spatial bottleneck (d_model ↔ bottleneck_dim)
        if bottleneck_dim is not None:
            self.encoder_proj = nn.Linear(d_model, bottleneck_dim)
            self.decoder_proj = nn.Linear(bottleneck_dim, d_model)
        else:
            self.encoder_proj = None
            self.decoder_proj = None

        # Temporal bottleneck (learnable stride compression/expansion)
        dim = bottleneck_dim if bottleneck_dim is not None else d_model
        if temporal_stride > 1:
            self.temporal_enc = nn.Conv1d(
                dim, dim,
                kernel_size=temporal_stride,
                stride=temporal_stride,
                padding=0,
            )
            self.temporal_dec = nn.ConvTranspose1d(
                dim, dim,
                kernel_size=temporal_stride,
                stride=temporal_stride,
                padding=0,
            )
        else:
            self.temporal_enc = None
            self.temporal_dec = None

    def encode(self, x):
        """
        Encode waveform → compressed latent.

        Pipeline: waveform → AudioEncoder → (spatial proj) → (temporal downsample)

        Returns:
            z: (batch, T_compressed, dim)
               T_compressed = T / temporal_stride
        """
        z = self.encoder(x)                          # (B, T, d_model)
        if self.encoder_proj is not None:
            z = self.encoder_proj(z)                 # (B, T, bottleneck_dim)
        if self.temporal_enc is not None:
            z = z.transpose(1, 2)                    # (B, dim, T)
            z = self.temporal_enc(z)                 # (B, dim, T/stride)
            z = z.transpose(1, 2)                    # (B, T/stride, dim)
        return z

    def decode(self, z):
        """
        Decode compressed latent → waveform.

        Pipeline: (temporal upsample) → (spatial proj) → AudioDecoder → waveform
        """
        if self.temporal_dec is not None:
            z = z.transpose(1, 2)                    # (B, dim, T/stride)
            z = self.temporal_dec(z)                 # (B, dim, T)
            z = z.transpose(1, 2)                    # (B, T, dim)
        if self.decoder_proj is not None:
            z = self.decoder_proj(z)                 # (B, T, d_model)
        return self.decoder(z)

    def forward(self, x):
        """
        Args:
            x: (batch, 1, time) raw audio waveform
        Returns:
            (batch, 1, time) reconstructed audio
        """
        z = self.encode(x)
        return self.decode(z)
