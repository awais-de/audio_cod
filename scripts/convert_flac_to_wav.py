#!/usr/bin/env python3
"""
Convert all FLAC files in test-clean to WAV format using soundfile.
This avoids needing torchcodec/FFmpeg dependencies.
"""

import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from paths import get_dataset_paths

def convert_flac_to_wav(input_dir, output_dir=None):
    """Convert all FLAC files in input_dir to WAV format."""
    
    input_path = Path(input_dir)
    if output_dir is None:
        output_path = input_path.parent / (input_path.name + "_wav")
    else:
        output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all FLAC files
    flac_files = sorted(list(input_path.glob('**/*.flac')))
    print(f"Found {len(flac_files)} FLAC files")
    
    if not flac_files:
        print("No FLAC files found!")
        return
    
    successful = 0
    failed = 0
    
    for flac_file in tqdm(flac_files, desc='Converting FLAC to WAV'):
        try:
            # Read FLAC file
            data, sr = sf.read(str(flac_file))
            
            # Create output file path (preserve directory structure)
            rel_path = flac_file.relative_to(input_path)
            wav_file = output_path / rel_path.with_suffix('.wav')
            wav_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write WAV file
            sf.write(str(wav_file), data, sr)
            successful += 1
            
        except Exception as e:
            failed += 1
            print(f"Error converting {flac_file}: {e}")
    
    print(f"\n✅ Conversion complete: {successful} successful, {failed} failed")
    print(f"Output directory: {output_path}")
    return output_path

if __name__ == '__main__':
    input_dir = str(get_dataset_paths()["test_clean"])
    output_dir = convert_flac_to_wav(input_dir)
