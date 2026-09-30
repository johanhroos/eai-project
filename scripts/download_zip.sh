#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
DL_DIR="$DATA_DIR/downloads"

mkdir -p "$DL_DIR"

echo "Downloading vMAP archive (~45 GB)..."

wget \
    --continue \
    --show-progress \
    --progress=bar:force:noscroll \
    --directory-prefix "$DL_DIR" \
    "https://huggingface.co/datasets/kxic/vMAP/resolve/main/vmap.zip"

echo
echo "Done."
echo "Archive: $DL_DIR/vmap.zip"