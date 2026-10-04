#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../remote/environment.sh"
MODEL=${1:?Usage: serve.sh /path/to/exported/chat/model}
exec "$PRETRAIN_ROOT/envs/inference/bin/vllm" serve "$MODEL" \
  --host 127.0.0.1 --port 8000 --served-model-name dummym-english \
  --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 2048 \
  --generation-config vllm --gpu-memory-utilization 0.5
