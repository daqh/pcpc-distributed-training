#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/cluster-env.sh"

LOCAL_PROJECT_DIR="$(pwd)/smallaudio"
REMOTE_PROJECT_DIR="~/smallaudio"

gcloud compute ssh cpu-train-node-0 \
  --zone="$ZONE" \
  --quiet \
  --command="
    set -e

    cd $REMOTE_PROJECT_DIR
    source .venv/bin/activate

    sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 7860

    tmux kill-session -t train 2>/dev/null || true
    tmux kill-session -t tensorboard 2>/dev/null || true

    tmux new-session -d -s gradio '
      cd $REMOTE_PROJECT_DIR &&
      source .venv/bin/activate &&
      pip install gradio &&
      python run-gradio.py
      exec bash
    '
  "
