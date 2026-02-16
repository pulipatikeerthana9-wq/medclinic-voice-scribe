#!/bin/bash
# MedClinic Quick Start Setup Script
# Run: bash setup.sh

set -e

echo "🏥 MedClinic Setup"
echo "=================="
echo ""

# Check Python version
echo "✓ Checking Python..."
python_version=$(python3 --version 2>&1)
echo "  Found: $python_version"

# Create virtual environment
echo "✓ Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "✓ Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the server:"
echo "  source venv/bin/activate"
echo "  python main.py"
echo ""
echo "Then open: http://localhost:8000"
echo ""
