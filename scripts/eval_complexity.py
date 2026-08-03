"""Result 19 — Complexity & latency comparison: ours (Phase G) vs EnCodec 24kHz.

Reports, for a 1-second input clip at each model's native sample rate:
  - trainable parameter count (encoder / decoder / bottleneck split)
  - codebook / buffer storage (EnCodec RVQ has none of this in ours)
  - MACs (thop) for encode and decode passes
  - wall-clock encode+decode latency (CPU, mean of N runs)
  - algorithmic delay (receptive field / first-frame latency), computed analytically
"""
import time
import torch
from thop import profile
from src.model import NeuralAudioCodec


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def count_buffers(module):
    return sum(b.numel() for b in module.buffers())


def load_ours(ckpt_path="checkpoints_active/temporal_phaseG/best.pt"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = NeuralAudioCodec(
        sample_rate=16000,
        hop_length=160,
        d_model=ckpt["d_model"],
        n_layers=ckpt["n_layers"],
        n_heads=ckpt["n_heads"],
        window_size=ckpt["window_size"],
        bottleneck_dim=ckpt["bottleneck_dim"],
        temporal_stride=ckpt["temporal_stride"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def ours_breakdown(model):
    enc = count_params(model.encoder)
    dec = count_params(model.decoder)
    proj = count_params(model.encoder_proj) + count_params(model.decoder_proj)
    temporal = count_params(model.temporal_enc) + count_params(model.temporal_dec)
    total = count_params(model)
    buffers = count_buffers(model)
    return {"encoder": enc, "decoder": dec, "bottleneck_proj": proj,
            "temporal_stride": temporal, "total": total, "buffers": buffers}


def encodec_breakdown(m):
    enc = count_params(m.encoder)
    dec = count_params(m.decoder)
    quant = count_params(m.quantizer)
    total = count_params(m)
    buffers = count_buffers(m)
    return {"encoder": enc, "decoder": dec, "quantizer": quant,
             "total": total, "buffers": buffers}


def time_call(fn, *args, n_warmup=3, n_runs=10):
    for _ in range(n_warmup):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn(*args)
    return (time.perf_counter() - t0) / n_runs


def min_samples_for_one_frame(encode_fn, sample_rate, lo=1, hi=None):
    """Binary search: fewest raw samples needed to produce >=1 output frame.
    This is the architecture's algorithmic (buffering) delay, independent of
    compute time and of any chunking choice made by the inference script."""
    hi = hi or sample_rate
    with torch.no_grad():
        while lo < hi:
            mid = (lo + hi) // 2
            x = torch.randn(1, 1, mid)
            try:
                n_out = encode_fn(x)
                if n_out >= 1:
                    hi = mid
                else:
                    lo = mid + 1
            except Exception:
                lo = mid + 1
    return lo


if __name__ == "__main__":
    torch.manual_seed(0)

    # ---- Ours ----
    ours = load_ours()
    ob = ours_breakdown(ours)
    x16 = torch.randn(1, 1, 16000)  # 1s @ 16kHz

    with torch.no_grad():
        macs_enc_ours, _ = profile(ours, inputs=(x16,), verbose=False)

    t_encode_ours = time_call(lambda: ours.encode(x16))
    with torch.no_grad():
        z_ours = ours.encode(x16)
    t_decode_ours = time_call(lambda: ours.decode(z_ours))

    print("=== Ours (Phase G, 16 kHz, causal, 3-bit SQ) ===")
    for k, v in ob.items():
        print(f"  {k:16s} {v:>14,}")
    print(f"  MACs (1s clip, full fwd) {macs_enc_ours:>14,.0f}")
    print(f"  encode latency (CPU)     {t_encode_ours*1000:>10.2f} ms")
    print(f"  decode latency (CPU)     {t_decode_ours*1000:>10.2f} ms")

    min_ours = min_samples_for_one_frame(
        lambda x: ours.encode(x).shape[1], sample_rate=16000
    )
    print(f"  algorithmic delay (min buffering for 1 output frame): "
          f"{min_ours} samples = {min_ours/16000*1000:.2f} ms @ 16kHz")
    print(f"  current encode.py/infer_offline.py default chunking: "
          f"1000.00 ms (--chunk-sec 1.0, configurable)")

    # ---- EnCodec 24kHz @ 6.0 kbps ----
    from encodec import EncodecModel
    enc_model = EncodecModel.encodec_model_24khz()
    enc_model.set_target_bandwidth(6.0)
    enc_model.eval()
    eb = encodec_breakdown(enc_model)

    x24 = torch.randn(1, 1, 24000)  # 1s @ 24kHz

    with torch.no_grad():
        macs_full_enc, _ = profile(enc_model, inputs=(x24,), verbose=False)

    def encodec_encode():
        with torch.no_grad():
            return enc_model.encode(x24)

    def encodec_decode(frames):
        with torch.no_grad():
            return enc_model.decode(frames)

    t_encode_enc = time_call(encodec_encode)
    frames = encodec_encode()
    t_decode_enc = time_call(lambda: encodec_decode(frames))

    print("\n=== EnCodec 24kHz (6.0 kbps target) ===")
    for k, v in eb.items():
        print(f"  {k:16s} {v:>14,}")
    print(f"  MACs (1s clip, full fwd) {macs_full_enc:>14,.0f}")
    print(f"  encode latency (CPU)     {t_encode_enc*1000:>10.2f} ms")
    print(f"  decode latency (CPU)     {t_decode_enc*1000:>10.2f} ms")
    print(f"  frame_rate               {enc_model.frame_rate} Hz "
          f"(hop = {1000/enc_model.frame_rate:.2f} ms)")
    print("  NOTE: released 24kHz checkpoint uses symmetric/reflect padding "
          "(causal=None) — it is NOT the paper's causal streaming variant, "
          "so a locally measured algorithmic delay would not be a fair "
          "like-for-like number. Use the paper's reported frame hop (13.3 ms) "
          "as the reference for EnCodec's streaming configuration instead.")

    print(f"\nTotal params  — ours: {ob['total']:,} | EnCodec: {eb['total']:,} "
          f"(ratio {eb['total']/ob['total']:.2f}x)")
    print(f"Buffer storage — ours: {ob['buffers']:,} | EnCodec: {eb['buffers']:,} "
          f"(ratio {eb['buffers']/max(ob['buffers'],1):.2f}x)")
