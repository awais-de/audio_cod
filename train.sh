#!/bin/bash
# Quick start script for Neural Audio Codec training

set -e

echo ""
echo "╔═════════════════════════════════════════════════════════════════════════════╗"
echo "║           Neural Audio Codec - Training Startup Script                      ║"
echo "╚═════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade requirements
echo "📚 Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "🔍 Running sanity check..."
python scripts/sanity_check.py

echo ""
echo "🚀 Starting training..."
python src/train.py
