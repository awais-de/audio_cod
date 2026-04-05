#!/bin/bash
# Setup script for 2-PC demo
# Installs required system dependencies for audio I/O

echo "================================"
echo "Audio Codec Demo Setup"
echo "================================"
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux"
    echo "Installing PortAudio..."
    
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        sudo apt-get update
        sudo apt-get install -y portaudio19-dev python3-pyaudio
    elif command -v yum &> /dev/null; then
        # RedHat/CentOS
        sudo yum install -y portaudio-devel
    elif command -v pacman &> /dev/null; then
        # Arch
        sudo pacman -S portaudio
    else
        echo "⚠️  Could not detect package manager"
        echo "Please install PortAudio manually:"
        echo "  - Debian/Ubuntu: sudo apt-get install portaudio19-dev"
        echo "  - RedHat/CentOS: sudo yum install portaudio-devel"
        echo "  - Arch: sudo pacman -S portaudio"
    fi
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS"
    echo "Installing PortAudio..."
    
    if command -v brew &> /dev/null; then
        brew install portaudio
    else
        echo "⚠️  Homebrew not found"
        echo "Please install Homebrew first: https://brew.sh"
        echo "Then run: brew install portaudio"
    fi
    
else
    echo "⚠️  Unsupported OS: $OSTYPE"
    echo "Please install PortAudio manually"
fi

echo ""
echo "Installing Python dependencies..."
./venv/bin/pip install sounddevice

echo ""
echo "✅ Setup complete!"
echo ""
echo "Test audio devices with:"
echo "  ./venv/bin/python -c 'import sounddevice as sd; print(sd.query_devices())'"
echo ""
echo "Run demo with:"
echo "  Server: python scripts/demo_server.py"
echo "  Client: python scripts/demo_client.py --host <server_ip>"
