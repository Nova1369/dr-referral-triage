#!/usr/bin/env bash
# Download the frozen ImageNet ResNet-50 backbone (timm release v0.1-weights, 102 MB) and check it.
# It is too large for the repository, so it is linked here instead.
set -euo pipefail
URL="https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnet50_ram-a26f946b.pth"
SHA="a26f946bdcb2e618263d03d8b60b66a26f7788dbadc1940c315b2bf62020a3e7"
OUT="${1:-checkpoints/resnet50_ram-a26f946b.pth}"
mkdir -p "$(dirname "$OUT")"
[ -f "$OUT" ] || curl -L --fail -o "$OUT" "$URL"
if command -v sha256sum >/dev/null; then GOT=$(sha256sum "$OUT" | cut -d' ' -f1); else GOT=$(shasum -a 256 "$OUT" | cut -d' ' -f1); fi
[ "$GOT" = "$SHA" ] && echo "OK: $OUT (sha256 matches)" || { echo "sha256 mismatch: $GOT"; exit 1; }
