#!/usr/bin/env python3
"""
Real-time Audio Codec Demo - Client (Receiver)
Receives encoded audio from server, decodes with neural codec, plays through speakers
"""

import socket
import struct
import torch
import sounddevice as sd
import numpy as np
import argparse
import queue
import threading
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.model import NeuralAudioCodec

class AudioClient:
    def __init__(self, checkpoint_path, server_host, server_port):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            server_host: Server IP address
            server_port: Server port
        """
        self.server_host = server_host
        self.server_port = server_port
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
        
        # Audio buffer
        self.audio_queue = queue.Queue(maxsize=10)
        self.running = False
        
    def decode_chunk(self, encoded_data, latent_shape):
        """Decode latent representation back to audio"""
        with torch.no_grad():
            # Convert bytes back to tensor
            latent_np = np.frombuffer(encoded_data, dtype=np.float16).reshape(latent_shape)
            latent = torch.from_numpy(latent_np).float().to(self.device)
            
            # Decode
            audio = self.model.decoder(latent)
            
            # Convert to numpy
            audio_np = audio.squeeze().cpu().numpy()
            return audio_np
    
    def receive_thread(self, sock):
        """Thread that receives and decodes audio data"""
        try:
            while self.running:
                # Receive size
                size_data = sock.recv(4)
                if not size_data:
                    print("Connection closed by server")
                    break
                
                size = struct.unpack('!I', size_data)[0]
                
                # Receive encoded data
                encoded_data = b''
                while len(encoded_data) < size:
                    chunk = sock.recv(size - len(encoded_data))
                    if not chunk:
                        break
                    encoded_data += chunk
                
                if len(encoded_data) != size:
                    print("Incomplete data received")
                    break
                
                # Decode (latent shape is [1, d_model, seq_len])
                # We need to infer the shape from the data size
                # For d_model=256, each timestep is 256*2 bytes (fp16) = 512 bytes
                d_model = 256
                seq_len = size // (d_model * 2)
                latent_shape = (1, d_model, seq_len)
                
                audio = self.decode_chunk(encoded_data, latent_shape)
                
                # Add to playback queue
                try:
                    self.audio_queue.put_nowait(audio)
                except queue.Full:
                    # Drop frame if buffer is full (prevents latency buildup)
                    try:
                        self.audio_queue.get_nowait()
                        self.audio_queue.put_nowait(audio)
                    except:
                        pass
                        
        except Exception as e:
            print(f"Error in receive thread: {e}")
        finally:
            self.running = False
    
    def audio_callback(self, outdata, frames, time, status):
        """Callback for audio output stream"""
        if status:
            print(f"Status: {status}")
        
        try:
            # Get audio from queue
            audio = self.audio_queue.get_nowait()
            
            # Pad or trim to match required frames
            if len(audio) < frames:
                audio = np.pad(audio, (0, frames - len(audio)))
            else:
                audio = audio[:frames]
            
            outdata[:] = audio.reshape(-1, 1)
            
        except queue.Empty:
            # No audio available, output silence
            outdata.fill(0)
    
    def run(self):
        """Connect to server and start playback"""
        print(f"\n{'='*70}")
        print(f"🔊 Audio Client")
        print(f"{'='*70}")
        print(f"Connecting to {self.server_host}:{self.server_port}...")
        
        try:
            # Connect to server
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.server_host, self.server_port))
            print(f"✅ Connected to server\n")
            
            self.running = True
            
            # Start receive thread
            receive_thread = threading.Thread(target=self.receive_thread, args=(sock,))
            receive_thread.daemon = True
            receive_thread.start()
            
            # Start audio playback
            with sd.OutputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=320,  # 20ms at 16kHz
                callback=self.audio_callback
            ):
                print("🔊 PLAYING (Press Ctrl+C to stop)")
                print("-" * 70)
                print(f"Listening to audio from {self.server_host}...")
                print(f"Queue size: {self.audio_queue.qsize()}/10")
                print()
                
                # Monitor queue status
                import time
                last_queue_size = 0
                while self.running:
                    time.sleep(0.5)
                    queue_size = self.audio_queue.qsize()
                    if queue_size != last_queue_size:
                        print(f"\rQueue: {queue_size}/10 ", end='', flush=True)
                        last_queue_size = queue_size
                    
                    if not receive_thread.is_alive():
                        break
            
            print("\n\n🛑 Playback stopped")
            
        except KeyboardInterrupt:
            print("\n\n🛑 Client stopped")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.running = False
            if 'sock' in locals():
                sock.close()

def main():
    parser = argparse.ArgumentParser(description='Neural Audio Codec - Streaming Client')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--host', type=str, required=True,
                       help='Server IP address')
    parser.add_argument('--port', type=int, default=9999,
                       help='Server port (default: 9999)')
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    # Create and run client
    client = AudioClient(args.checkpoint, args.host, args.port)
    client.run()

if __name__ == '__main__':
    main()
