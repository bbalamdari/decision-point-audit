#!/usr/bin/env bash
# Download the exact framework releases and tokenizers used in the paper (Section 5) into the repository root.
# The wheels are only unpacked and read (single modules imported by file path), never installed.
# Gemma 4 is a gated model on Hugging Face: accept its licence and log in (huggingface-cli login) first.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dl_pkgs dl_src dl_gemma4 dl_qwen3
for pkg in axolotl==0.21.1 llamafactory==0.9.5 trl==1.17.1 unsloth==2026.10.11 unsloth_zoo==2026.10.11; do
  python -m pip download --no-deps --only-binary=:all: -d dl_pkgs "$pkg"
done
for whl in dl_pkgs/*.whl; do
  name=$(basename "$whl" .whl)
  python -m zipfile -e "$whl" "dl_src/${name%%-py3*}-py3-none-any"
done
python - <<'EOF'
from huggingface_hub import hf_hub_download
for f in ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json"):
    hf_hub_download("google/gemma-4-E2B-it", f, local_dir="dl_gemma4")
for f in ("merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"):
    hf_hub_download("Qwen/Qwen3-0.6B", f, local_dir="dl_qwen3")
EOF
sha256sum dl_pkgs/*.whl dl_gemma4/chat_template.jinja dl_gemma4/tokenizer.json dl_qwen3/tokenizer_config.json
