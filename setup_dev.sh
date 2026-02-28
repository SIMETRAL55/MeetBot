#!/bin/bash
# Quick setup script for MeetBot development environment

set -e

echo "==========================================="
echo "MeetBot Development Environment Setup"
echo "==========================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  No virtual environment detected!"
    echo "Please activate your virtual environment first:"
    echo "  source venv/bin/activate"
    echo ""
    echo "Or create one:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate"
    exit 1
fi

echo "✓ Virtual environment: $VIRTUAL_ENV"

# Install in development mode
echo ""
echo "Installing MeetBot in development mode..."
cd "$SCRIPT_DIR"
pip install -e .

echo ""
echo "==========================================="
echo "✓ Setup Complete!"
echo "==========================================="
echo ""
echo "You can now use MeetBot commands:"
echo ""
echo "  # Transcribe audio"
echo "  python -m meetbot.cli transcribe audio.m4a --language ja --backend local"
echo ""
echo "  # Or use the convenient alias:"
echo "  meetbot transcribe audio.m4a --language ja --backend local"
echo ""
echo "  # Build index"
echo "  meetbot index results/audio.json"
echo ""
echo "  # Ask questions"
echo "  meetbot query db/audio 'Your question here'"
echo ""
