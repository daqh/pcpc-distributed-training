#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/cluster-env.sh"

wait_for_ssh() {
  local instance="$1"
  local zone="$2"

  echo "Waiting for SSH on $instance..."

  until gcloud compute ssh "$instance" \
    --zone="$zone" \
    --quiet \
    --command="echo SSH is ready" >/dev/null 2>&1
  do
    echo "SSH not ready yet for $instance, retrying..."
    sleep 5
  done

  echo "SSH ready on $instance"
}

for i in $(seq 0 $((NUM_NODES - 1)));
do
  INSTANCE="cpu-train-node-$i"

  if [ "$i" -eq 0 ]; then
    INSTANCE_TAGS="$TAG,http-server,https-server"
  else
    INSTANCE_TAGS="$TAG"
  fi

  gcloud compute instances create "$INSTANCE" \
    --zone="$ZONE" \
    --machine-type=c4-highmem-8 \
    --boot-disk-size=100GB \
    --image-family=ubuntu-2404-lts-amd64 \
    --image-project=ubuntu-os-cloud \
    --network="$NETWORK" \
    --tags="$INSTANCE_TAGS"

  wait_for_ssh "$INSTANCE" "$ZONE"

  gcloud compute ssh "$INSTANCE" \
    --zone="$ZONE" \
    --quiet \
    --command="
      set -e

      echo 'Aggiorno apt...'
      sudo apt-get update

      sudo apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        git \
        htop \
        tmux \
        cmake \
        ninja-build \
        build-essential \
        python3-dev \
        libopenmpi-dev \
        openmpi-bin
    "
done

# Permette ai nodi del cluster di comunicare tra loro sulla porta master.
gcloud compute firewall-rules create allow-cpu-train-ddp-master \
  --network="$NETWORK" \
  --allow=tcp,udp,icmp \
  --source-ranges=10.0.0.0/8 \
  --target-tags="$TAG" \
  --source-tags="$TAG" || true

gcloud compute firewall-rules create allow-tensorboard-6006-public \
  --network="$NETWORK" \
  --allow=tcp:6006 \
  --source-ranges=0.0.0.0/0 \
  --target-tags="$TAG" || true
