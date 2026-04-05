#!/usr/bin/env python3
"""
V4 Fine-tuning - Direct continuation from V3
Uses V3 checkpoint directly and continues training
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import NeuralAudioCodec

# Configuration for V4 continuation
learning_rate = 5e-7  # Very conservative - continue from V3
n_epochs = 15  # Continue for more epochs
n_dataset_files = 2000  # Use more files
init_checkpoint = "checkpoints_emergency/pesq_balanced_v3_20260129_094112/best.pt"

SAMPLE_RATE = 16000
WINDOW_SIZE = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRAD_CLIP = 1.0

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"pesq_v4_continuation_{timestamp}"
checkpoint_dir = Path("checkpoints_emergency") / run_name
checkpoint_dir.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*80}")
print(f"V4 CONTINUATION TRAINING")
print(f"{'='*80}")
print(f"\nInit from: {init_checkpoint}")
print(f"Learning Rate: {learning_rate}")
print(f"Epochs: {n_epochs}")
print(f"Dataset: {n_dataset_files} files")
print(f"Run Name: {run_name}\n")

# Dataset
class AudioDataset(Dataset):
    def __init__(self, data_dir, n_files=None, sample_rate=16000, window_size=384):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.window_size = window_size
        
        self.files = sorted(self.data_dir.glob("**/*.flac")) + sorted(self.data_dir.glob("**/*.wav"))
        if n_files:
            self.files = self.files[:n_files]
        
        print(f"Dataset: {len(self.files)} files loaded")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio_path = self.files[idx]
        waveform, sr = torchaudio.load(str(audio_path))
        
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        if waveform.abs().max() > 0:
            waveform = waveform / (waveform.abs().max() + 1e-8)
        
        if waveform.size(-1) < self.window_size:
            waveform = torch.nn.functional.pad(waveform, (0, self.window_size - waveform.size(-1)))
        else:
            waveform = waveform[:, :self.window_size]
        
        return waveform.squeeze(0)

def compute_stft_loss(output, target, n_fft=512, hop_length=128):
    window = torch.hann_window(n_fft, device=output.device)
    output_stft = torch.stft(output, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    target_stft = torch.stft(target, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    output_mag = torch.abs(output_stft)
    target_mag = torch.abs(target_stft)
    return torch.mean(torch.abs(output_mag - target_mag))

def main():
    print(f"Device: {DEVICE}\n")
    
    # Load model
    model = NeuralAudioCodec().to(DEVICE)
    
    # Load V3 checkpoint
    if Path(init_checkpoint).exists():
        print(f"Loading V3 checkpoint...")
        state = torch.load(init_checkpoint, map_location=DEVICE)
        
        # Handle different checkpoint formats
        if isinstance(state, dict):
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            # Filter out non-model keys
            state = {k: v for k, v in state.items() if k.startswith(('encoder', 'decoder'))}
        
        model.load_state_dict(state, strict=False)
        print(f"✓ V3 checkpoint loaded\n")
    else:
        print(f"ERROR: V3 checkpoint not found")
        return
    
    # Dataset
    dataset = AudioDataset("datasets", n_files=n_dataset_files, sample_rate=SAMPLE_RATE, window_size=WINDOW_SIZE)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Training
    optimizer = Adam(model.parameters(), lr=learning_rate)
    mse_loss = MSELoss()
    best_loss = float('inf')
    no_improve_count = 0
    
    print(f"{'='*80}")
    print(f"STARTING V4 TRAINING")
    print(f"{'='*80}\n")
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, audio in enumerate(dataloader):
            audio = audio.to(DEVICE).unsqueeze(1)
            output = model(audio)
            
            mse = mse_loss(output, audio)
            stft = compute_stft_loss(output.squeeze(1), audio.squeeze(1))
            loss = 2.0 * stft + 0.5 * mse
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        progress = (epoch + 1) / n_epochs * 100
        print(f"Epoch {epoch+1:2d}/{n_epochs} | Loss: {avg_loss:.6f} | Progress: {progress:5.1f}%")
        
        # Save if improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve_count = 0
            checkpoint_path = checkpoint_dir / "best.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Best checkpoint saved")
        else:
            no_improve_count += 1
            if no_improve_count >= 3:
                print(f"\n✓ Early stopping after {epoch+1} epochs")
                break
    
    print(f"\n{'='*80}")
    print(f"V4 TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"\nCheckpoint saved to: {checkpoint_dir}/best.pt")
    print(f"Run: {run_name}")

if __name__ == "__main__":
    main()
