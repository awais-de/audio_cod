#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

p = Path('/Users/muhammadawais/Downloads/ac_proj/audio_cod/results/compare_fair_bitrate.csv')
df = pd.read_csv(p)

print("="*80)
print("FAIR BITRATE COMPARISON (Target: 10 kbps)")
print("="*80)
print(f"\nNEURAL CODEC (with 1-bit quantization):")
print(f"  Bitrate achieved (mean): {df['neural_bitrate_achieved_bps'].mean()/1000:.2f} kbps")
print(f"  Bitrate target: {df['neural_bitrate_target_kbps'].iloc[0]} kbps")
print(f"  Latency (avg): {df['neural_latency_ms_avg'].mean():.2f} ms")
print(f"  PESQ (mean): {df['neural_pesq'].mean():.3f}")
print(f"  STOI (mean): {df['neural_stoi'].mean():.3f}")

print(f"\nAAC CODEC:")
print(f"  Bitrate achieved (mean): {df['ffmpeg_bitrate_achieved_bps'].mean()/1000:.2f} kbps")
print(f"  Bitrate target: {df['ffmpeg_bitrate_target_kbps'].iloc[0]} kbps")
print(f"  PESQ (mean): {df['ffmpeg_pesq'].mean():.3f}")
print(f"  STOI (mean): {df['ffmpeg_stoi'].mean():.3f}")

print(f"\n{'='*80}")
print("COMPARISON:")
print(f"Neural is {df['neural_bitrate_achieved_bps'].mean() / df['ffmpeg_bitrate_achieved_bps'].mean():.1f}x higher bitrate than AAC")
print(f"AAC PESQ is {df['ffmpeg_pesq'].mean() / df['neural_pesq'].mean():.1f}x higher than Neural")
print(f"Neural STOI is {df['neural_stoi'].mean() / df['ffmpeg_stoi'].mean():.1f}x higher than AAC")
print(f"{'='*80}")
