#!/usr/bin/env python3
"""
Real-time Audio Codec Demo - Server (Sender)
Captures audio from microphone, encodes with neural codec, streams over network
"""

import socket
import struct
import torch
import torchaudio
import sounddevice as sd
import numpy as np
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.model import NeuralAudioCodec

class AudioServer:
    def __init__(self, checkpoint_path, host='0.0.0.0', port=9999, chunk_size=512):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            host: Host to bind to (0.0.0.0 for all interfaces)
            port: Port to listen on
            chunk_size: Audio chunk size in samples (512 = 32ms at 16kHz)
        """
        self.host = host
        self.port = port
        self.chunk_size = chunk_size
        self.sample_rate = 16000
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = NeuralAudioCodec(
            d_model=checkpoint.get('d_model', 256),
            n_layers=checkpoint.get('n_layers', 4),
            n_heads=checkpoint.get('n_heads', 8)
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✅ Model loaded on {self.device}")
        
        self.resampler = torchaudio.transforms.Resample(44100, 16000)
        
    def encode_chunk(self, audio_chunk):
        """Encode audio chunk and return compressed latent representation"""
        with torch.no_grad():
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0).to(self.device)
            
            # Normalize
            audio_tensor = audio_tensor / (audio_tensor.abs().max() + 1e-8)
            
            # Encode only (get latent representation)
            latent = self.model.encoder(audio_tensor)
            
            # Convert to bytes (compress to fp16 to save bandwidth)
            latent_np = latent.cpu().half().numpy()
            return latent_np.tobytes()
    
    def audio_callback(self, indata, frames, time, status, client_socket):
        """Callback for audio input stream"""
        if status:
            print(f"Status: {status}")
        
        try:
            # Convert to mono if stereo
            if indata.shape[1] > 1:
                audio = indata[:, 0]
            else:
                audio = indata.flatten()
            
            # Resample to 16kHz if needed
            if self.sample_rate != 44100:
                audio = audio
            else:
                audio_tensor = torch.from_numpy(audio).float()
                audio = self.resampler(audio_tensor).numpy()
            
            # Encode
            encoded_data = self.encode_chunk(audio)
            
            # Send: [size (4 bytes)][data]
            size = len(encoded_data)
            client_socket.sendall(struct.pack('!I', size))
            client_socket.sendall(encoded_data)
            
        except Exception as e:
            print(f"Error in callback: {e}")
    
    def run(self):
        """Start server and stream audio"""
        # Create socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(1)
        
        print(f"\n{'='*70}")
        print(f"🎤 Audio Server Ready")
        print(f"{'='*70}")
        print(f"Listening on {self.host}:{self.port}")
        print(f"Chunk size: {self.chunk_size} samples ({self.chunk_size/self.sample_rate*1000:.1f}ms)")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Waiting for client connection...")
        print()
        
        while True:
            try:
                # Wait for client
                client_socket, client_address = server_socket.accept()
                print(f"✅ Client connected: {client_address}")
                print(f"Starting audio capture...\n")
                
                # Start audio stream
                with sd.InputStream(
                    channels=1,
                    samplerate=self.sample_rate,
                    blocksize=self.chunk_size,
                    callback=lambda *args: self.audio_callback(*args, client_socket=client_socket)
                ):
                    print("🔴 STREAMING (Press Ctrl+C to stop)")
                    print("-" * 70)
                    print("Speak into your microphone...")
                    print()
                    
                    # Keep streaming until client disconnects or error
                    try:
                        while True:
                            # Check if client is still connected
                            ready = client_socket.recv(1, socket.MSG_PEEK)
                            if not ready:
                                break
                            sd.sleep(100)
                    except:
                        pass
                
                print("\n❌ Client disconnected")
                client_socket.close()
                print("Waiting for new connection...\n")
                
            except KeyboardInterrupt:
                print("\n\n🛑 Server stopped")
                break
            except Exception as e:
                print(f"Error: {e}")
                if 'client_socket' in locals():
                    client_socket.close()

def main():
    parser = argparse.ArgumentParser(description='Neural Audio Codec - Streaming Server')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to (default: 0.0.0.0 for all interfaces)')
    parser.add_argument('--port', type=int, default=9999,
                       help='Port to listen on (default: 9999)')
    parser.add_argument('--chunk-size', type=int, default=320,
                       help='Audio chunk size in samples (default: 320 = 20ms at 16kHz)')
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    # Create and run server
    server = AudioServer(args.checkpoint, args.host, args.port, args.chunk_size)
    server.run()

if __name__ == '__main__':
    main()
