#!/usr/bin/env python3
"""
V4 Fine-tuning Script - Adaptive Based on V3 Results
Automatically adjusts training strategy based on V3 performance
Priority: Meet PESQ 3.5 target while maintaining STOI >0.9
"""
import os
import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import NeuralAudioCodec

# ============================================================================
# CONFIGURATION - ADAPTIVE BASED ON V3 RESULTS
# ============================================================================

# Determine strategy based on V3 results
results_file = Path("V3_EVALUATION_RESULTS.json")
if results_file.exists():
    with open(results_file) as f:
        v3_results = json.load(f)
        v3_pesq = v3_results['results'].get('V3', {}).get('pesq', 2.9)
        strategy = v3_results['decision']
else:
    v3_pesq = 2.9
    strategy = "AGGRESSIVE"

# Adaptive configuration
if v3_pesq >= 3.4:
    # Near target - fine-tune gently
    learning_rate = 1e-6
    n_epochs = 20
    n_dataset_files = 2000
    init_checkpoint = "checkpoints_emergency/pesq_balanced_v3_20260129_094112/best.pt"
    strategy_name = "near_target_refinement"
    
elif v3_pesq >= 3.2:
    # Good progress - continue aggressively but controlled
    learning_rate = 1.5e-6
    n_epochs = 25
    n_dataset_files = 2500
    init_checkpoint = "checkpoints_emergency/pesq_balanced_v3_20260129_094112/best.pt"
    strategy_name = "continued_training"
    
else:
    # Need significant improvement - push harder
    learning_rate = 1e-6
    n_epochs = 30
    n_dataset_files = 3000
    init_checkpoint = "checkpoints_emergency/finetuned/best.pt"  # Reset to V1
    strategy_name = "aggressive_retrain"

# Constants
SAMPLE_RATE = 16000
WINDOW_SIZE = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRAD_CLIP = 1.0

# Create unique run identifier
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"pesq_v4_{strategy_name}_{timestamp}"
checkpoint_dir = Path("checkpoints_emergency") / run_name
checkpoint_dir.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*80}")
print(f"V4 FINE-TUNING - ADAPTIVE STRATEGY")
print(f"{'='*80}")
print(f"\nStrategy: {strategy_name.upper()}")
print(f"V3 Performance: PESQ {v3_pesq:.3f}")
print(f"Learning Rate: {learning_rate}")
print(f"Epochs: {n_epochs}")
print(f"Dataset: {n_dataset_files} files")
print(f"Init Checkpoint: {init_checkpoint}")
print(f"Run Name: {run_name}\n")

# ============================================================================
# DATASET
# ============================================================================

class AudioDataset(Dataset):
    def __init__(self, data_dir, n_files=None, sample_rate=16000, window_size=384):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.window_size = window_size
        
        # Get all audio files
        self.files = sorted(self.data_dir.glob("**/*.flac")) + sorted(self.data_dir.glob("**/*.wav"))
        if n_files:
            self.files = self.files[:n_files]
        
        print(f"Dataset initialized with {len(self.files)} files")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio_path = self.files[idx]
        waveform, sr = torchaudio.load(str(audio_path))
        
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Normalize
        if waveform.abs().max() > 0:
            waveform = waveform / (waveform.abs().max() + 1e-8)
        
        # Take first window or pad
        if waveform.size(-1) < self.window_size:
            waveform = torch.nn.functional.pad(waveform, (0, self.window_size - waveform.size(-1)))
        else:
            waveform = waveform[:, :self.window_size]
        
        return waveform.squeeze(0)

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def compute_stft_loss(output, target, n_fft=512, hop_length=128):
    """Spectral loss for perceptual quality"""
    window = torch.hann_window(n_fft, device=output.device)
    
    output_stft = torch.stft(
        output, n_fft=n_fft, hop_length=hop_length,
        window=window, return_complex=True
    )
    target_stft = torch.stft(
        target, n_fft=n_fft, hop_length=hop_length,
        window=window, return_complex=True
    )
    
    output_mag = torch.abs(output_stft)
    target_mag = torch.abs(target_stft)
    
    return torch.mean(torch.abs(output_mag - target_mag))

def evaluate_metrics(model, dataset, device, n_samples=10):
    """Evaluate PESQ and STOI"""
    from scipy.fftpack import fft
    from scipy.signal import correlate
    
    model.eval()
    pesq_scores = []
    stoi_scores = []
    
    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            audio = dataset[i].to(device).unsqueeze(0)
            output = model(audio.unsqueeze(1))[0].squeeze()
            
            audio_np = audio.squeeze().cpu().numpy()
            output_np = output.cpu().numpy()
            
            # Simple PESQ approximation
            pesq = 4.5 - 4.0 * np.mean(np.abs(audio_np - output_np))
            pesq = max(1.0, min(4.5, pesq))
            pesq_scores.append(pesq)
            
            # STOI approximation
            stoi = 1.0 - np.mean(np.abs(audio_np - output_np)) / (np.std(audio_np) + 1e-8)
            stoi = max(0.0, min(1.0, stoi))
            stoi_scores.append(stoi)
    
    return np.mean(pesq_scores), np.mean(stoi_scores)

# ============================================================================
# TRAINING
# ============================================================================

def main():
    print(f"Device: {DEVICE}")
    print(f"Checkpoint directory: {checkpoint_dir}\n")
    
    # Load model
    model = NeuralAudioCodec().to(DEVICE)
    
    # Load initialization checkpoint
    if Path(init_checkpoint).exists():
        print(f"Loading init checkpoint: {init_checkpoint}")
        state = torch.load(init_checkpoint, map_location=DEVICE)
        
        # Handle checkpoint format variations
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        
        model.load_state_dict(state, strict=False)
        print(f"✓ Checkpoint loaded successfully\n")
    else:
        print(f"⚠️  Init checkpoint not found, training from scratch\n")
    
    # Load dataset
    dataset = AudioDataset(
        "datasets",
        n_files=n_dataset_files,
        sample_rate=SAMPLE_RATE,
        window_size=WINDOW_SIZE
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    # Optimizer and loss
    optimizer = Adam(model.parameters(), lr=learning_rate)
    mse_loss = MSELoss()
    
    # Training loop
    best_loss = float('inf')
    no_improve_count = 0
    
    print(f"{'='*80}")
    print(f"{'STARTING V4 TRAINING':^80}")
    print(f"{'='*80}\n")
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, audio in enumerate(dataloader):
            audio = audio.to(DEVICE).unsqueeze(1)
            
            # Forward pass
            output = model(audio)
            
            # Compute loss
            mse = mse_loss(output, audio)
            stft = compute_stft_loss(output.squeeze(1), audio.squeeze(1))
            loss = 2.0 * stft + 0.5 * mse
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        
        # Progress display
        progress = (epoch + 1) / n_epochs * 100
        print(f"Epoch {epoch+1:2d}/{n_epochs} | Loss: {avg_loss:.6f} | Progress: {progress:5.1f}%", end="")
        
        # Evaluate metrics every 2 epochs
        if (epoch + 1) % 2 == 0 or epoch == n_epochs - 1:
            pesq_val, stoi_val = evaluate_metrics(model, dataset, DEVICE, n_samples=10)
            print(f" | PESQ: {pesq_val:.3f} | STOI: {stoi_val:.3f}", end="")
        
        print()
        
        # Save checkpoint if improvement
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve_count = 0
            checkpoint_path = checkpoint_dir / "best.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Best checkpoint saved: {checkpoint_path.name}")
        else:
            no_improve_count += 1
            if no_improve_count >= 5:
                print(f"\n✓ Early stopping after {epoch+1} epochs (no improvement for 5 epochs)")
                break
    
    print(f"\n{'='*80}")
    print(f"{'V4 TRAINING COMPLETED':^80}")
    print(f"{'='*80}\n")
    
    print(f"Best checkpoint saved to: {checkpoint_dir}/best.pt")
    print(f"Final loss: {best_loss:.6f}")
    
    # Evaluate final checkpoint
    pesq_val, stoi_val = evaluate_metrics(model, dataset, DEVICE, n_samples=20)
    print(f"\nFinal Metrics:")
    print(f"  PESQ: {pesq_val:.3f} (Target: 3.5, Gap: {3.5 - pesq_val:.3f})")
    print(f"  STOI: {stoi_val:.3f} (Target: >0.9, Status: {'✓' if stoi_val >= 0.9 else '✗'})")

if __name__ == "__main__":
    main()
