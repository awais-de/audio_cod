#!/usr/bin/env python3
"""
File-based Audio Codec Demo
Simpler alternative to real-time streaming - uses pre-recorded audio files
"""

import socket
import struct
import torch
import torchaudio
import soundfile as sf
import argparse
from pathlib import Path
import time
import sys
import numpy as np
sys.path.append(str(Path(__file__).parent.parent))
from src.model import NeuralAudioCodec

def load_model(checkpoint_path, device):
    """Load neural codec model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = NeuralAudioCodec(
        d_model=checkpoint.get('d_model', 256),
        n_layers=checkpoint.get('n_layers', 4),
        n_heads=checkpoint.get('n_heads', 8)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def run_server(checkpoint_path, audio_file, host, port, chunk_ms=20):
    """Stream audio file over network"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on {device}...")
    model = load_model(checkpoint_path, device)
    
    # Load audio
    print(f"Loading audio from {audio_file}...")
    audio_np, sr = sf.read(audio_file)
    audio = torch.from_numpy(audio_np).float()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    else:
        audio = audio.T  # soundfile returns (samples, channels), we need (channels, samples)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    
    # Ensure 2D shape [1, samples]
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    print(f"Audio: {audio.shape[1]/16000:.2f}s, {audio.shape[1]} samples, shape={audio.shape}")
    
    # Create chunks
    chunk_size = int(16000 * chunk_ms / 1000)
    num_chunks = audio.shape[1] // chunk_size
    print(f"Chunk size: {chunk_size} samples ({chunk_ms}ms)")
    print(f"Total chunks: {num_chunks}")
    
    # Start server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)
    
    print(f"\n{'='*70}")
    print(f"📡 File-based Audio Server")
    print(f"{'='*70}")
    print(f"Listening on {host}:{port}")
    print(f"Waiting for client...\n")
    
    client_socket, client_addr = server_socket.accept()
    print(f"✅ Client connected: {client_addr}\n")
    print("🔴 STREAMING")
    print("-" * 70)
    
    try:
        with torch.no_grad():
            for i in range(num_chunks):
                # Extract chunk
                start = i * chunk_size
                end = start + chunk_size
                chunk = audio[:, start:end].to(device)
                
                # Encode
                latent = model.encoder(chunk)
                
                # Send
                latent_bytes = latent.cpu().half().numpy().tobytes()
                size = len(latent_bytes)
                client_socket.sendall(struct.pack('!I', size))
                client_socket.sendall(latent_bytes)
                
                # Progress
                if i % 50 == 0:
                    progress = (i / num_chunks) * 100
                    elapsed = (i * chunk_ms) / 1000
                    print(f"  Chunk {i}/{num_chunks} ({progress:.1f}%) - {elapsed:.2f}s", end='\r')
                
                # Maintain real-time pacing
                time.sleep(chunk_ms / 1000)
        
        print(f"\n\n✅ Streaming complete ({num_chunks} chunks)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        client_socket.close()
        server_socket.close()

def run_client(checkpoint_path, output_file, host, port):
    """Receive and decode audio stream"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on {device}...")
    model = load_model(checkpoint_path, device)
    
    print(f"\n{'='*70}")
    print(f"📥 File-based Audio Client")
    print(f"{'='*70}")
    print(f"Connecting to {host}:{port}...\n")
    
    # Connect
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    print(f"✅ Connected\n")
    print("🔊 RECEIVING")
    print("-" * 70)
    
    audio_chunks = []
    chunk_count = 0
    
    try:
        with torch.no_grad():
            while True:
                # Receive size
                size_data = sock.recv(4)
                if not size_data:
                    break
                
                size = struct.unpack('!I', size_data)[0]
                
                # Receive data
                encoded_data = b''
                while len(encoded_data) < size:
                    chunk = sock.recv(size - len(encoded_data))
                    if not chunk:
                        break
                    encoded_data += chunk
                
                if len(encoded_data) != size:
                    break
                
                # Decode
                d_model = 256
                seq_len = size // (d_model * 2)
                latent_shape = (1, d_model, seq_len)
                latent_np = np.frombuffer(encoded_data, dtype=np.float16).reshape(latent_shape)
                latent = torch.from_numpy(latent_np).float().to(device)
                
                audio = model.decoder(latent)
                audio_chunks.append(audio.cpu())
                
                chunk_count += 1
                if chunk_count % 50 == 0:
                    print(f"  Received {chunk_count} chunks", end='\r')
        
        print(f"\n\n✅ Received {chunk_count} chunks")
        
        # Save audio
        if audio_chunks:
            print(f"Saving to {output_file}...")
            full_audio = torch.cat(audio_chunks, dim=-1)
            torchaudio.save(output_file, full_audio, 16000)
            print(f"✅ Saved {full_audio.shape[1]/16000:.2f}s of audio")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        sock.close()

def main():
    parser = argparse.ArgumentParser(description='File-based Audio Codec Demo')
    parser.add_argument('--mode', type=str, required=True, choices=['server', 'client'],
                       help='Run as server or client')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Server hostname/IP')
    parser.add_argument('--port', type=int, default=9999,
                       help='Port number')
    parser.add_argument('--input', type=str,
                       help='Input audio file (server mode)')
    parser.add_argument('--output', type=str, default='received_audio.wav',
                       help='Output audio file (client mode)')
    parser.add_argument('--chunk-ms', type=int, default=20,
                       help='Chunk size in milliseconds (server mode)')
    
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    if args.mode == 'server':
        if not args.input:
            print("❌ --input required in server mode")
            return
        if not Path(args.input).exists():
            print(f"❌ Input file not found: {args.input}")
            return
        
        run_server(args.checkpoint, args.input, args.host, args.port, args.chunk_ms)
    
    else:  # client
        run_client(args.checkpoint, args.output, args.host, args.port)

if __name__ == '__main__':
    main()
