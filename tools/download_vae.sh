#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$SCRIPT_DIR/models"

mkdir -p "$TARGET_DIR"

echo "Downloading VAE model (CleanVAE) to $TARGET_DIR"

wget -L \
  -P "$TARGET_DIR" \
  "https://civitai.com/api/download/models/125553" \
  --content-disposition

echo "Download complete"
