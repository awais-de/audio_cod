"""
Low-Latency Neural Audio Coder with Transformer Architecture
Encoder-Decoder implementation for real-time speech compression
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
        # Remove future frames to maintain causality
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class CausalAttention(nn.Module):
    """Sliding-window causal attention for low latency"""
    def __init__(self, d_model, n_heads, window_size=512, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.d_k)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x)  # (batch, seq_len, 3*d_model)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, n_heads, seq_len, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, n_heads, seq_len, seq_len)
        
        # Create causal mask - prevent attention to future positions
        device = x.device
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        
        # Optional: Apply sliding window (limit attention to recent past)
        if self.window_size is not None and seq_len > self.window_size:
            window_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), 
                                     diagonal=self.window_size + 1).bool()
            causal_mask = causal_mask | window_mask
        
        # Apply mask by setting to very negative value (not -inf to avoid NaN)
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
        
        # Softmax with numerical stability
        attn = F.softmax(attn, dim=-1)
        
        # Replace any NaN with zeros
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch, n_heads, seq_len, d_k)
        out = out.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.d_model)
        
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with causal attention and feed-forward network"""
    def __init__(self, d_model, n_heads, d_ff=None, window_size=512, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
            
        self.attention = CausalAttention(d_model, n_heads, window_size, dropout)
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
        # Self-attention with residual
        x = x + self.attention(self.norm1(x))
        # Feed-forward with residual
        x = x + self.ffn(self.norm2(x))
        return x


class AudioEncoder(nn.Module):
    """
    Streaming audio encoder with causal convolutions and transformer layers.
    Compresses raw audio to latent representations with minimal latency.
    """
    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        hop_length=160,  # 10ms at 16kHz
        d_model=512,
        n_layers=8,
        n_heads=16,
        window_size=512,
        dropout=0.1
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.d_model = d_model
        
        # Downsampling: raw audio -> embeddings
        # Using strided causal convolutions to reduce temporal resolution
        self.conv_layers = nn.ModuleList([
            # Layer 1: (batch, 1, time) -> (batch, 64, time/4)
            nn.Sequential(
                CausalConv1d(1, 64, kernel_size=7, stride=2),
                nn.GroupNorm(8, 64),
                nn.GELU(),
            ),
            # Layer 2: (batch, 64, time/4) -> (batch, 128, time/8)
            nn.Sequential(
                CausalConv1d(64, 128, kernel_size=7, stride=2),
                nn.GroupNorm(16, 128),
                nn.GELU(),
            ),
            # Layer 3: (batch, 128, time/8) -> (batch, 256, time/16)
            nn.Sequential(
                CausalConv1d(128, 256, kernel_size=7, stride=2),
                nn.GroupNorm(32, 256),
                nn.GELU(),
            ),
            # Layer 4: (batch, 256, time/16) -> (batch, d_model, time/16)
            nn.Sequential(
                CausalConv1d(256, d_model, kernel_size=3, stride=1),
                nn.GroupNorm(32, d_model),
                nn.GELU(),
            ),
        ])
        
        # Transformer encoder layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, window_size=window_size, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: (batch, 1, time) raw audio waveform
        Returns:
            (batch, seq_len, d_model) encoded representations
        """
        # Downsampling through convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # (batch, d_model, seq_len) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2)
        
        # Transformer encoding
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        
        return x


class AudioDecoder(nn.Module):
    """
    Streaming audio decoder that reconstructs waveform from latent representations.
    Uses transformer layers followed by transposed convolutions for upsampling.
    """
    def __init__(
        self,
        sample_rate=16000,
        hop_length=160,
        d_model=512,
        n_layers=8,
        n_heads=16,
        window_size=512,
        dropout=0.1
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.d_model = d_model
        
        # Transformer decoder layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, window_size=window_size, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Upsampling: embeddings -> raw audio
        # Using transposed convolutions to increase temporal resolution
        self.deconv_layers = nn.ModuleList([
            # Layer 1: (batch, d_model, seq_len) -> (batch, 256, seq_len)
            nn.Sequential(
                nn.ConvTranspose1d(d_model, 256, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(32, 256),
                nn.GELU(),
            ),
            # Layer 2: (batch, 256, seq_len) -> (batch, 128, seq_len*2)
            nn.Sequential(
                nn.ConvTranspose1d(256, 128, kernel_size=8, stride=2, padding=3, output_padding=0),
                nn.GroupNorm(16, 128),
                nn.GELU(),
            ),
            # Layer 3: (batch, 128, seq_len*2) -> (batch, 64, seq_len*4)
            nn.Sequential(
                nn.ConvTranspose1d(128, 64, kernel_size=8, stride=2, padding=3, output_padding=0),
                nn.GroupNorm(8, 64),
                nn.GELU(),
            ),
            # Layer 4: (batch, 64, seq_len*4) -> (batch, 32, seq_len*8)
            nn.Sequential(
                nn.ConvTranspose1d(64, 32, kernel_size=8, stride=2, padding=3, output_padding=0),
                nn.GroupNorm(4, 32),
                nn.GELU(),
            ),
            # Final layer: (batch, 32, seq_len*8) -> (batch, 1, seq_len*8) with proper output padding
            nn.ConvTranspose1d(32, 1, kernel_size=7, stride=1, padding=3),
        ])
        
        # Output activation
        self.output_activation = nn.Tanh()
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model) encoded representations
        Returns:
            (batch, 1, time) reconstructed audio waveform
        """
        # Transformer decoding
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # (batch, seq_len, d_model) -> (batch, d_model, seq_len)
        x = x.transpose(1, 2)
        
        # Upsampling through deconvolutional layers
        for deconv_layer in self.deconv_layers:
            x = deconv_layer(x)
        
        # Apply output activation to constrain to [-1, 1]
        x = self.output_activation(x)
        
        return x


class NeuralAudioCodec(nn.Module):
    """
    Complete neural audio codec combining encoder and decoder.
    Provides interface for compression and reconstruction.
    """
    def __init__(
        self,
        sample_rate=16000,
        hop_length=160,
        d_model=512,
        n_layers=8,
        n_heads=16,
        window_size=512,
        dropout=0.1
    ):
        super().__init__()
        
        self.encoder = AudioEncoder(
            sample_rate=sample_rate,
            hop_length=hop_length,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            window_size=window_size,
            dropout=dropout
        )
        
        self.decoder = AudioDecoder(
            sample_rate=sample_rate,
            hop_length=hop_length,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            window_size=window_size,
            dropout=dropout
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, 1, time) raw audio waveform
        Returns:
            (batch, 1, time) reconstructed audio waveform
        """
        # Encode
        latent = self.encoder(x)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        return reconstructed
    
    def encode(self, x):
        """Encode audio to latent representation"""
        return self.encoder(x)
    
    def decode(self, latent):
        """Decode latent representation to audio"""
        return self.decoder(latent)
    
    def get_latency_ms(self):
        """Calculate theoretical latency in milliseconds"""
        # Encoder downsampling: 2*2*2 = 8x reduction
        # Each layer adds some latency due to kernel size and stride
        encoder_latency = (7 + 7 + 7 + 3) / 2  # Approximate kernel delay
        
        # Transformer is causal with sliding window, minimal added latency
        transformer_latency = 1  # Frame delay
        
        # Decoder upsampling
        decoder_latency = (3 + 8 + 8 + 8 + 7) / 2
        
        total_samples = encoder_latency + transformer_latency + decoder_latency
        latency_ms = (total_samples / self.encoder.sample_rate) * 1000
        
        return latency_ms


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = NeuralAudioCodec(
        sample_rate=16000,
        hop_length=160,
        d_model=512,
        n_layers=8,
        n_heads=16,
        window_size=512,
        dropout=0.1
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Estimated latency: {model.get_latency_ms():.2f} ms")
    
    # Test with random audio (1 second at 16kHz)
    batch_size = 2
    audio_length = 16000
    x = torch.randn(batch_size, 1, audio_length).to(device)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        # Encode
        latent = model.encode(x)
        print(f"Latent shape: {latent.shape}")
        
        # Decode
        reconstructed = model.decode(latent)
        print(f"Output shape: {reconstructed.shape}")
        
        # Full forward
        output = model(x)
        print(f"Full forward output shape: {output.shape}")
        
        # Calculate compression ratio
        input_size = x.numel() * 32  # 32-bit float
        latent_size = latent.numel() * 32
        compression_ratio = input_size / latent_size
        print(f"\nCompression ratio: {compression_ratio:.2f}x")
        
        # Calculate theoretical bitrate (assuming quantization to 8-bit)
        frames_per_second = 16000 / 160  # hop_length
        tokens_per_frame = latent.shape[1] / (audio_length / 160)
        bitrate_kbps = (tokens_per_frame * 512 * 8 * frames_per_second) / 1000
        print(f"Theoretical bitrate (before quantization): {bitrate_kbps:.2f} kbps")
