#!/usr/bin/env bash

ZONE="europe-west8-b"
NETWORK="default"
TAG="cpu-train-cluster"

NUM_NODES=3
PROCS_PER_NODE=3
MASTER_PORT=29500
TRAIN_SCRIPT="train.py"
