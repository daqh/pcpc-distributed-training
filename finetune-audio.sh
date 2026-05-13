#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/cluster-env.sh"

LOCAL_PROJECT_DIR="$(pwd)/smallaudio"
REMOTE_PROJECT_DIR="~/smallaudio"

install_project_dependencies() {
  local instance="$1"

  echo "Creo virtualenv e installo requirements su $instance..."

  gcloud compute ssh "$instance" \
    --zone="$ZONE" \
    --quiet \
    --command="
      set -e

      cd $REMOTE_PROJECT_DIR

      python3 -m venv .venv
      source .venv/bin/activate

      python -m pip install --upgrade pip setuptools wheel
      python -m pip install numpy pyyaml typing_extensions sympy filelock networkx jinja2
      python -m pip install cmake ninja

      if [ -f requirements.txt ]; then
        pip install --only-binary=:all: -r requirements.txt
      else
        echo 'requirements.txt non trovato in $REMOTE_PROJECT_DIR'
        exit 1
      fi

      pip install torch --index-url https://download.pytorch.org/whl/cpu
    "
}

for NODE_RANK in $(seq 0 $((NUM_NODES - 1)));
do
  INSTANCE="cpu-train-node-$NODE_RANK"

  echo "Copio il progetto locale su $INSTANCE..."

  gcloud compute scp \
    --zone="$ZONE" \
    --recurse \
    "$LOCAL_PROJECT_DIR" \
    "$INSTANCE:$REMOTE_PROJECT_DIR"

  install_project_dependencies "$INSTANCE"
done

MASTER_ADDR=$(gcloud compute instances describe cpu-train-node-0 \
  --zone="$ZONE" \
  --format='get(networkInterfaces[0].networkIP)')

echo "MASTER_ADDR=$MASTER_ADDR"

for NODE_RANK in $(seq 0 $((NUM_NODES - 1)));
do
  INSTANCE="cpu-train-node-$NODE_RANK"

  echo "Starting training on $INSTANCE with NODE_RANK=$NODE_RANK"

  gcloud compute ssh "$INSTANCE" \
    --zone="$ZONE" \
    --quiet \
    --command="
      set -e

      cd $REMOTE_PROJECT_DIR
      source .venv/bin/activate

      tmux kill-session -t train 2>/dev/null || true

      tmux new-session -d -s train '
        cd $REMOTE_PROJECT_DIR &&
        source .venv/bin/activate &&

        export MASTER_ADDR=$MASTER_ADDR &&
        export MASTER_PORT=$MASTER_PORT &&
        export NODE_RANK=$NODE_RANK &&
        export NUM_NODES=$NUM_NODES &&

        torchrun \
          --nnodes=$NUM_NODES \
          --nproc_per_node=$PROCS_PER_NODE \
          --node_rank=$NODE_RANK \
          --master_addr=$MASTER_ADDR \
          --master_port=$MASTER_PORT \
          $TRAIN_SCRIPT

      exec bash
      '
    "
done

gcloud compute ssh cpu-train-node-0 \
  --zone="$ZONE" \
  --quiet \
  --command="
    set -e

    cd $REMOTE_PROJECT_DIR
    source .venv/bin/activate

    sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 6006

    tmux kill-session -t tensorboard 2>/dev/null || true

    tmux new-session -d -s tensorboard '
      cd $REMOTE_PROJECT_DIR &&
      source .venv/bin/activate &&
      pip install tensorboard &&
      tensorboard \
        --logdir lightning_logs \
        --bind_all \
        --port 6006

    exec bash
    '
  "
