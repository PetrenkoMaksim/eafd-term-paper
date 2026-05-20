#!/usr/bin/env bash
set -euo pipefail

apt-get update
apt-get install -y \
    git git-lfs curl wget unzip zip htop tmux tree rsync \
    build-essential software-properties-common \
    openjdk-17-jre-headless \
    python3 python3-pip python3-venv

git lfs install || true

mkdir -p /workspace/venvs

nvidia-smi || true
