"""
Baseline Codec Wrapper (Opus)
For comparison with neural codec in project demo
"""

import numpy as np
import subprocess
import tempfile
import os
from pathlib import Path


class OpusCodec:
    """Wrapper for Opus codec (industry standard for VoIP)"""
    
    def __init__(self, bitrate=16000, sample_rate=16000):
        """
        Initialize Opus codec
        
        Args:
            bitrate: Target bitrate in bps (e.g., 16000 for 16 kbps)
            sample_rate: Audio sample rate in Hz
        """
        self.bitrate = bitrate
        self.sample_rate = sample_rate
        self.check_opus_installed()
    
    def check_opus_installed(self):
        """Check if opus tools are installed"""
        try:
            subprocess.run(['opusenc', '--version'], 
                          capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  Opus tools not found. Install with:")
            print("   sudo apt-get install opus-tools")
            print("   Or on Mac: brew install opus-tools")
            return False
    
    def encode_decode(self, audio, bitrate=None):
        """
        Encode and decode audio through Opus codec
        
        Args:
            audio: numpy array of audio samples (mono, float32, -1 to 1)
            bitrate: Override default bitrate (optional)
            
        Returns:
            reconstructed: numpy array of reconstructed audio
            actual_bitrate: actual bitrate used
        """
        if bitrate is None:
            bitrate = self.bitrate
        
        # Create temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            input_wav = os.path.join(tmpdir, 'input.wav')
            output_opus = os.path.join(tmpdir, 'output.opus')
            output_wav = os.path.join(tmpdir, 'output.wav')
            
            # Save input as WAV
            self._save_wav(input_wav, audio)
            
            # Encode to Opus
            subprocess.run([
                'opusenc',
                '--bitrate', str(bitrate // 1000),  # Convert to kbps
                '--raw-rate', str(self.sample_rate),
                input_wav,
                output_opus
            ], capture_output=True, check=True)
            
            # Decode from Opus
            subprocess.run([
                'opusdec',
                output_opus,
                output_wav
            ], capture_output=True, check=True)
            
            # Load reconstructed audio
            reconstructed = self._load_wav(output_wav)
            
            # Calculate actual bitrate
            opus_size = os.path.getsize(output_opus)
            duration = len(audio) / self.sample_rate
            actual_bitrate = (opus_size * 8) / duration
            
            return reconstructed, actual_bitrate
    
    def _save_wav(self, filepath, audio):
        """Save audio as WAV file"""
        import soundfile as sf
        # Ensure audio is in correct range
        audio = np.clip(audio, -1.0, 1.0)
        sf.write(filepath, audio, self.sample_rate, subtype='PCM_16')
    
    def _load_wav(self, filepath):
        """Load audio from WAV file"""
        import soundfile as sf
        audio, sr = sf.read(filepath)
        assert sr == self.sample_rate, f"Sample rate mismatch: {sr} vs {self.sample_rate}"
        return audio


def main():
    """Test Opus codec"""
    import soundfile as sf
    
    print("=" * 80)
    print("OPUS BASELINE CODEC TEST")
    print("=" * 80)
    print()
    
    # Check if opus is installed
    codec = OpusCodec(bitrate=16000, sample_rate=16000)
    
    # Load test audio
    test_audio = "/mnt/Data/muaw1874/datasets/LibriSpeech/train-clean-100/2007/149877/2007-149877-0049.flac"
    
    if not Path(test_audio).exists():
        print(f"❌ Test audio not found: {test_audio}")
        return
    
    print(f"📄 Loading test audio: {Path(test_audio).name}")
    audio, sr = sf.read(test_audio)
    
    if sr != 16000:
        print(f"   Resampling from {sr} Hz to 16000 Hz")
        audio = np.interp(
            np.linspace(0, len(audio), int(len(audio) * 16000 / sr)),
            np.arange(len(audio)),
            audio
        )
    
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Limit to 10 seconds
    audio = audio[:10 * 16000]
    
    print(f"   Duration: {len(audio) / 16000:.2f}s")
    print()
    
    # Test different bitrates
    bitrates = [8000, 12000, 16000, 24000, 32000]
    
    print("Testing Opus at different bitrates:")
    print("-" * 80)
    
    for br in bitrates:
        print(f"🔊 Bitrate: {br // 1000} kbps")
        
        try:
            reconstructed, actual_br = codec.encode_decode(audio, bitrate=br)
            
            # Calculate SNR
            min_len = min(len(audio), len(reconstructed))
            audio_aligned = audio[:min_len]
            recon_aligned = reconstructed[:min_len]
            
            noise = audio_aligned - recon_aligned
            signal_power = np.mean(audio_aligned ** 2)
            noise_power = np.mean(noise ** 2)
            
            if noise_power > 1e-10:
                snr = 10 * np.log10(signal_power / noise_power)
            else:
                snr = 100.0
            
            print(f"   Actual bitrate: {actual_br / 1000:.2f} kbps")
            print(f"   SNR: {snr:.2f} dB")
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print()
    
    print("=" * 80)
    print("✅ Opus codec test complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
