#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/environment.sh"
cd "$PRETRAIN_ROOT/repo"
curl --fail --location --retry 3 --max-time 30 --head https://huggingface.co
# uv 放在项目目录，既不修改系统 Python，也不修改其他项目的 Conda 环境。
if [[ ! -x "$PRETRAIN_ROOT/tools/uv" ]]; then
  curl --fail --location --retry 3 https://github.com/astral-sh/uv/releases/download/0.8.22/uv-x86_64-unknown-linux-gnu.tar.gz -o "$TMPDIR/uv.tar.gz"
  tar -xzf "$TMPDIR/uv.tar.gz" -C "$PRETRAIN_ROOT/tools" --strip-components=1
fi
UV="$PRETRAIN_ROOT/tools/uv"
if [[ ! -x "$PRETRAIN_ROOT/envs/train/bin/python" ]]; then
  "$UV" venv --python 3.11 "$PRETRAIN_ROOT/envs/train"
fi
PYTHON="$PRETRAIN_ROOT/envs/train/bin/python"
"$UV" pip install --python "$PYTHON" torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
"$UV" pip install --python "$PYTHON" -e '.[train,data,dev,workflow]'
"$UV" pip freeze --python "$PYTHON" > "$PRETRAIN_ROOT/envs/train.lock.txt"
"$PYTHON" -c 'import torch; print(torch.__version__, torch.cuda.device_count()); assert torch.cuda.device_count() == 4; assert torch.cuda.is_bf16_supported()'
