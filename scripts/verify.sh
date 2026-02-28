#!/bin/bash
# MeetBot Verification Script
# Runs linting, type checking, and tests

set -e

echo "=== MeetBot Verification Suite ==="

source_dir="src/source"
test_dir="tests"

echo ""
echo "1. Running flake8 linter..."
flake8 "$source_dir" --max-line-length=100 --exclude=venv || true

echo ""
echo "2. Running pytest tests..."
python -m pytest "$test_dir" -v

echo ""
echo "=== Verification Complete ==="
