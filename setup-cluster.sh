#!/usr/bin/env bash

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

set -e

ZONE="europe-west8-b"
NETWORK="default"
TAG="cpu-train-cluster"

MASTER_PORT="29500"

LOCAL_PROJECT_DIR="$(pwd)/smallm"
REMOTE_PROJECT_DIR="~/smallm"

for i in $(seq 0 2);
do
  INSTANCE="cpu-train-node-$i"

  # If i == 0
  if [ "$i" -eq 0 ]; then
    gcloud compute instances create $INSTANCE \
      --zone=$ZONE \
      --machine-type=c4-standard-4 \
      --boot-disk-size=100GB \
      --image-family=ubuntu-2404-lts-amd64 \
      --image-project=ubuntu-os-cloud \
      --network=$NETWORK \
      --tags=$TAG,http-server,https-server
  else
    gcloud compute instances create $INSTANCE \
      --zone=$ZONE \
      --machine-type=c4-standard-4 \
      --boot-disk-size=100GB \
      --image-family=ubuntu-2404-lts-amd64 \
      --image-project=ubuntu-os-cloud \
      --network=$NETWORK \
      --tags=$TAG
  fi

  wait_for_ssh $INSTANCE $ZONE
  
  gcloud compute ssh $INSTANCE \
    --zone=$ZONE \
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
        tmux

      sudo apt install -y \
        git cmake ninja-build build-essential \
        python3-dev \
        libopenmpi-dev openmpi-bin
    "

  echo "Copio il progetto locale su $INSTANCE..."

  gcloud compute scp \
    --zone="$ZONE" \
    --recurse \
    "$LOCAL_PROJECT_DIR" \
    "$INSTANCE:$REMOTE_PROJECT_DIR"

  echo "Creo virtualenv e installo requirements su $INSTANCE..."

  gcloud compute ssh "$INSTANCE" \
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

      # git clone --recursive https://github.com/pytorch/pytorch
      # cd pytorch
      # git checkout v2.9.0
      # git submodule sync
      # git submodule update --init --recursive
      # export USE_CUDA=0
      # export USE_ROCM=0
      # export USE_MPI=1
      # export CMAKE_PREFIX_PATH=\"$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')\"
      # python setup.py develop
      pip install torch --index-url https://download.pytorch.org/whl/cpu
    "
done

# Permette ai nodi del cluster di comunicare tra loro sulla porta master.
gcloud compute firewall-rules create allow-cpu-train-ddp-master \
  --network="$NETWORK" \
  --allow=tcp,udp,icmp \
  --source-ranges=10.0.0.0/8 \
  --target-tags="$TAG" \
  --source-tags="$TAG"

gcloud compute firewall-rules create allow-tensorboard-6006-public \
  --network="$NETWORK" \
  --allow=tcp:6006 \
  --source-ranges=0.0.0.0/0 \
  --target-tags="$TAG" \
  --source-tags="$TAG"

