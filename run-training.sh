#!/usr/bin/env bash
set -e

ZONE="europe-west8-b"
NUM_NODES=3
PROCS_PER_NODE=3
MASTER_PORT=29500
REMOTE_PROJECT_DIR="~/smallm"
TRAIN_SCRIPT="train.py"

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
  --zone=europe-west8-b \
  --quiet \
  --command="
    set -e

    cd ~/smallm
    source .venv/bin/activate

    sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 6006

    tmux kill-session -t tensorboard 2>/dev/null || true

    tmux new-session -d -s tensorboard '
      cd ~/smallm &&
      pip install tensorboard &&
      source .venv/bin/activate &&
      tensorboard \
        --logdir lightning_logs \
        --bind_all \
        --port 6006

    exec bash
    '
  "

gcloud compute firewall-rules create allow-tensorboard-6006 \
  --direction=INGRESS \
  --action=ALLOW \
  --rules=tcp:6006 \
  --source-ranges=0.0.0.0/32 \
  --target-tags="default"

