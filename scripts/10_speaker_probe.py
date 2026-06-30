#!/usr/bin/env python3
"""
Speaker Identity Linear Probe
==============================
Tests whether the Phase G encoder's latent space encodes speaker identity.

Method: extract mean-pooled latent vectors from all test-clean speakers,
train a multinomial logistic regression on a 70/30 split, report accuracy.

If the encoder has learned speaker-invariant representations (as would be
ideal for a codec), accuracy should be near chance (1/N_speakers).
If it encodes identity (as expected from MSE/spectral training without
adversarial disentanglement), accuracy should be significantly above chance.

Output: comparisons/YYYY-MM-DD_speaker_probe/
  report.txt   — accuracy, confusion summary, per-speaker recall
  features.npz — latent vectors + labels (for downstream PCA/t-SNE)
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import get_dataset_paths
from src.codec_utils import load_model, find_checkpoint

SR       = 16000
CLIP_SEC = 3        # shorter clips → more samples per speaker
MAX_CLIPS_PER_SPEAKER = 25


def extract_latent(model, audio: np.ndarray, device) -> np.ndarray:
    """Encode audio → mean-pool over time → (bottleneck_dim,) vector."""
    x = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        z = model.encode(x)          # (1, T, dim)
    return z.squeeze(0).mean(0).cpu().numpy()   # (dim,)


def main():
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = find_checkpoint(PROJECT_ROOT)

    timestamp = datetime.now().strftime('%Y-%m-%d')
    out_dir   = PROJECT_ROOT / 'comparisons' / f'{timestamp}_speaker_probe'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*68}")
    print("SPEAKER IDENTITY PROBE — linear classifier on frozen encoder")
    print(f"{'='*68}")
    print(f"checkpoint: {ckpt_path.parent.name}/{ckpt_path.name}")
    print(f"device    : {device}")
    print(f"clip_sec  : {CLIP_SEC}s  |  max {MAX_CLIPS_PER_SPEAKER} clips/speaker\n")

    model, _ = load_model(ckpt_path, device)
    model.eval()

    paths = get_dataset_paths()

    # Collect utterances per speaker
    speaker_files: dict[str, list] = {}
    for f in sorted(paths['test_clean'].rglob('*.flac')):
        spk = f.parts[-3]
        speaker_files.setdefault(spk, []).append(f)

    speakers = sorted(speaker_files.keys())
    n_spk    = len(speakers)
    spk2idx  = {s: i for i, s in enumerate(speakers)}
    print(f"Found {n_spk} speakers in test-clean\n")

    # Extract features
    X, y = [], []
    for spk in speakers:
        files = speaker_files[spk][:MAX_CLIPS_PER_SPEAKER]
        n_ok  = 0
        for fpath in files:
            audio, fsr = sf.read(fpath)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if fsr != SR:
                n = int(len(audio) * SR / fsr)
                audio = np.interp(np.linspace(0, len(audio), n), np.arange(len(audio)), audio)
            audio = np.clip(audio[:CLIP_SEC * SR], -1.0, 1.0).astype(np.float32)
            if len(audio) < 160:
                continue
            feat = extract_latent(model, audio, device)
            X.append(feat)
            y.append(spk2idx[spk])
            n_ok += 1
        print(f"  {spk} : {n_ok} clips")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"\nFeature matrix: {X.shape}  (samples × latent_dim)")
    print(f"Chance accuracy: {100.0 / n_spk:.1f}%\n")

    # Train/test split (stratified by speaker)
    np.random.seed(42)
    train_idx, test_idx = [], []
    for spk_i in range(n_spk):
        idx = np.where(y == spk_i)[0]
        np.random.shuffle(idx)
        split = max(1, int(len(idx) * 0.7))
        train_idx.extend(idx[:split].tolist())
        test_idx.extend(idx[split:].tolist())

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # Standardise
    mu    = X_train.mean(0, keepdims=True)
    sigma = X_train.std(0, keepdims=True) + 1e-8
    X_train = (X_train - mu) / sigma
    X_test  = (X_test  - mu) / sigma

    # Logistic regression
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, confusion_matrix
        clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                                 multi_class='multinomial')
        clf.fit(X_train, y_train)
        y_pred   = clf.predict(X_test)
        acc      = accuracy_score(y_test, y_pred) * 100
        cm       = confusion_matrix(y_test, y_pred)
        per_spk  = cm.diagonal() / cm.sum(axis=1).clip(1)
        sklearn_ok = True
        print(f"Logistic regression accuracy: {acc:.1f}%  (chance: {100.0/n_spk:.1f}%)")
    except ImportError:
        print("sklearn not available — install scikit-learn for logistic regression")
        sklearn_ok = False
        acc, per_spk = float('nan'), np.full(n_spk, float('nan'))

    # Save features for downstream PCA/t-SNE
    np.savez(out_dir / 'features.npz', X=X, y=y,
             speakers=np.array(speakers), mu=mu, sigma=sigma)

    # Report
    SEP = '=' * 68
    sep = '-' * 68
    lines = [
        '', SEP,
        'SPEAKER IDENTITY PROBE — linear logistic regression on frozen encoder',
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Checkpoint: {ckpt_path.parent.name}",
        f"Clips: {len(X)} total  ({CLIP_SEC}s each)  |  {n_spk} speakers",
        f"Train/test: {len(train_idx)} / {len(test_idx)}",
        SEP, '',
        f"Logistic regression accuracy : {acc:.1f}%",
        f"Chance (1/{n_spk})           : {100.0/n_spk:.1f}%",
        f"Above-chance ratio           : {acc / (100.0/n_spk):.2f}x",
        '',
        'Interpretation:',
        '  High accuracy => encoder preserves speaker identity (expected for codec)',
        '  Near-chance   => encoder has learned speaker-invariant representations',
        sep, '',
    ]
    if sklearn_ok:
        lines += [
            'PER-SPEAKER RECALL', sep,
            f"  {'Speaker':<12} {'Recall':>8} {'N_test':>8}",
            sep,
        ]
        for i, spk in enumerate(speakers):
            n_test = int(cm[i].sum())
            lines.append(f"  {spk:<12} {per_spk[i]*100:>7.1f}%  {n_test:>7}")
    lines += [sep, '',
              f"Features saved: {out_dir}/features.npz",
              '  (X: latent vectors, y: speaker indices, speakers: speaker IDs)',
              SEP]

    report = '\n'.join(lines)
    print(report)
    with open(out_dir / 'report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nreport:   {out_dir}/report.txt")
    print(f"features: {out_dir}/features.npz")


if __name__ == '__main__':
    main()
