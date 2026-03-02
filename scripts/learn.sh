#!/usr/bin/env bash
set -euo pipefail

echo "MeetBot learner quickstart"
echo "1) Open docs/learners/learning_path.md"
echo "2) Start with Module 0"

echo "\nQuick environment check:"
python --version || true
if [ -d .venv ]; then
  echo ".venv found"
else
  echo "No .venv yet (create with: python -m venv .venv)"
fi

echo "\nFirst file: docs/learners/learning_path.md"
