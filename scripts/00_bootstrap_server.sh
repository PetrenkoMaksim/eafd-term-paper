#!/usr/bin/env bash
set -euo pipefail

apt-get update
apt-get install -y \
    git git-lfs curl wget unzip zip htop tmux tree rsync \
    build-essential software-properties-common \
    openjdk-17-jre-headless \
    python3-pip

git lfs install || true

# micromamba: быстрее и чище, чем системный conda
if [ ! -d "$HOME/micromamba" ]; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /tmp bin/micromamba
    mkdir -p "$HOME/micromamba/bin"
    mv /tmp/bin/micromamba "$HOME/micromamba/bin/"
fi

echo 'export PATH="$HOME/micromamba/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/micromamba/bin:$PATH"

micromamba shell init -s bash -r "$HOME/micromamba" || true

nvidia-smi
