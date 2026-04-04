#!/usr/bin/env python3
"""
Generate comparison summary from evaluation results
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load results
neural_new = pd.read_csv('results/eval_10kbps_final.csv')
neural_prev = pd.read_csv('results/compare_fair_bitrate_phase3b.csv')

print("="*80)
print("COMPARISON SUMMARY: Neural Models vs AAC @ 10 kbps")
print("="*80)

# AAC stats from previous evaluation (from compare_fair_bitrate_phase3b.csv)
aac_bitrate = neural_prev['ffmpeg_bitrate_achieved_bps'].mean() / 1000
aac_pesq = neural_prev['ffmpeg_pesq'].mean()
aac_stoi = neural_prev['ffmpeg_stoi'].mean()

# Previous neural model stats
neural_prev_bitrate = neural_prev['neural_bitrate_achieved_bps'].mean() / 1000
neural_prev_pesq = neural_prev['neural_pesq'].mean()
neural_prev_stoi = neural_prev['neural_stoi'].mean()
neural_prev_latency = neural_prev['neural_latency_ms_avg'].mean()

# New neural model stats
neural_new_bitrate = neural_new['bitrate_achieved_bps'].mean() / 1000
neural_new_pesq = neural_new['pesq'].mean()
neural_new_stoi = neural_new['stoi'].mean()
neural_new_latency = neural_new['latency_ms_avg'].mean()

print(f"\n{'Metric':<30} {'AAC':<15} {'Neural (Prev)':<15} {'Neural (New)':<15}")
print("-"*80)
print(f"{'Target Bitrate (kbps)':<30} {'10.0':<15} {'10.0':<15} {'10.0':<15}")
print(f"{'Achieved Bitrate (kbps)':<30} {aac_bitrate:<15.2f} {neural_prev_bitrate:<15.2f} {neural_new_bitrate:<15.2f}")
print(f"{'Bitrate Overshoot (×)':<30} {aac_bitrate/10:<15.2f} {neural_prev_bitrate/10:<15.2f} {neural_new_bitrate/10:<15.2f}")
print(f"{'PESQ (mean)':<30} {aac_pesq:<15.3f} {neural_prev_pesq:<15.3f} {neural_new_pesq:<15.3f}")
print(f"{'STOI (mean)':<30} {aac_stoi:<15.3f} {neural_prev_stoi:<15.3f} {neural_new_stoi:<15.3f}")
print(f"{'Latency (ms)':<30} {'N/A':<15} {neural_prev_latency:<15.1f} {neural_new_latency:<15.1f}")

print("\n" + "="*80)
print("COMPARISON: Neural New vs Previous")
print("="*80)

bitrate_change = ((neural_new_bitrate - neural_prev_bitrate) / neural_prev_bitrate) * 100
pesq_change = ((neural_new_pesq - neural_prev_pesq) / neural_prev_pesq) * 100
stoi_change = ((neural_new_stoi - neural_prev_stoi) / neural_prev_stoi) * 100
latency_change = ((neural_new_latency - neural_prev_latency) / neural_prev_latency) * 100

print(f"Bitrate:    {neural_new_bitrate:6.2f} kbps → {neural_prev_bitrate:6.2f} kbps ({bitrate_change:+.1f}%)")
print(f"PESQ:       {neural_new_pesq:6.3f} → {neural_prev_pesq:6.3f} ({pesq_change:+.1f}%)")
print(f"STOI:       {neural_new_stoi:6.3f} → {neural_prev_stoi:6.3f} ({stoi_change:+.1f}%)")
print(f"Latency:    {neural_new_latency:6.1f} ms → {neural_prev_latency:6.1f} ms ({latency_change:+.1f}%)")

print("\n" + "="*80)
print("COMPARISON: Neural New vs AAC")
print("="*80)

bitrate_ratio = neural_new_bitrate / aac_bitrate
pesq_ratio = neural_new_pesq / aac_pesq
stoi_ratio = neural_new_stoi / aac_stoi

print(f"Bitrate:    Neural is {bitrate_ratio:.1f}× HIGHER than AAC ({neural_new_bitrate:.2f} vs {aac_bitrate:.2f} kbps)")
print(f"PESQ:       AAC is {1/pesq_ratio:.1f}× BETTER than Neural ({aac_pesq:.3f} vs {neural_new_pesq:.3f})")
print(f"STOI:       Neural is {stoi_ratio:.1f}× BETTER than AAC ({neural_new_stoi:.3f} vs {aac_stoi:.3f})")

print("\n" + "="*80)
print("VERDICT")
print("="*80)

print("\nNeural Model (New) Improvements:")
if pesq_change > 0:
    print(f"  ✓ Better perceptual quality (PESQ): +{pesq_change:.1f}%")
else:
    print(f"  ✗ Worse perceptual quality (PESQ): {pesq_change:.1f}%")

if stoi_change > 0:
    print(f"  ✓ Better intelligibility (STOI): +{stoi_change:.1f}%")
else:
    print(f"  ✗ Worse intelligibility (STOI): {stoi_change:.1f}%")

if abs(bitrate_change) < 5:
    print(f"  ≈ Similar bitrate: {bitrate_change:+.1f}%")
elif bitrate_change < 0:
    print(f"  ✓ Lower bitrate: {bitrate_change:.1f}%")
else:
    print(f"  ✗ Higher bitrate: +{bitrate_change:.1f}%")

print("\nNeural vs AAC:")
if neural_new_bitrate < aac_bitrate:
    print(f"  ✓ More efficient: {neural_new_bitrate:.1f} kbps vs {aac_bitrate:.1f} kbps")
else:
    print(f"  ✗ Less efficient: {neural_new_bitrate:.1f} kbps vs {aac_bitrate:.1f} kbps (×{bitrate_ratio:.1f})")

if neural_new_pesq > aac_pesq:
    print(f"  ✓ Better quality (PESQ): {neural_new_pesq:.3f} vs {aac_pesq:.3f}")
else:
    print(f"  ✗ Worse quality (PESQ): {neural_new_pesq:.3f} vs {aac_pesq:.3f}")

if neural_new_stoi > aac_stoi:
    print(f"  ✓ Better intelligibility (STOI): {neural_new_stoi:.3f} vs {aac_stoi:.3f}")
else:
    print(f"  ✗ Worse intelligibility (STOI): {neural_new_stoi:.3f} vs {aac_stoi:.3f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

print("""
The rate-distortion trained model (phase3b_l2_10kbps) shows:
  • 24% improvement in PESQ over previous phase3b model
  • 17% degradation in STOI over previous phase3b model
  • Similar bitrate inefficiency (~88 kbps, 8.8× over target)
  • Still significantly worse than AAC in both bitrate and PESQ

The model failed to achieve the 10 kbps target. The L2 norm rate proxy
reduced distortion but did not reduce entropy-coded bitrate. This suggests
the current approach (L2 penalty + 1-bit quantization) is insufficient for
achieving low-bitrate neural audio compression.

Recommendation: Fundamental redesign needed - consider vector quantization,
learned entropy models, or temporal compression techniques.
""")

print("="*80)
