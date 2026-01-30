#!/bin/bash
# Quick setup wrapper for Neural Audio Codec project
# Usage: ./setup.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================="
echo "Running Comprehensive Setup..."
echo "=================================="
echo ""

# Run the Python setup script
python3 "$PROJECT_ROOT/setup.py"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "✅ Setup Complete!"
    echo "=================================="
    echo ""
    echo "Next: Activate venv and run evaluation"
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        echo "  $PROJECT_ROOT/venv/Scripts/activate"
    else
        echo "  source $PROJECT_ROOT/venv/bin/activate"
    fi
else
    echo ""
    echo "❌ Setup failed with errors"
fi

exit $exit_code
