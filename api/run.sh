#!/usr/bin/env bash
# One-command launcher.
# Usage:  bash run.sh
set -e
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt
echo
echo "===================================================="
echo " Sheria-Bot NLP API running at:"
echo "   http://127.0.0.1:8000"
echo "   http://127.0.0.1:8000/docs   (Swagger UI)"
echo "===================================================="
uvicorn main:app --reload --port 8000
