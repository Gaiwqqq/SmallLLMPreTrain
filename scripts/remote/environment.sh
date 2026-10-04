#!/usr/bin/env bash
# source 此文件后，下载、编译及临时文件全部留在用户指定的训练根目录。
export PRETRAIN_ROOT=/diff/gaiwq/llm_pretrain
export PATH="$PRETRAIN_ROOT/envs/train/bin:$PATH"
export CONDA_PKGS_DIRS="$PRETRAIN_ROOT/cache/conda"
export PIP_CACHE_DIR="$PRETRAIN_ROOT/cache/pip"
export HF_HOME="$PRETRAIN_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$PRETRAIN_ROOT/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$PRETRAIN_ROOT/cache/torchinductor"
export TRITON_CACHE_DIR="$PRETRAIN_ROOT/cache/triton"
export CUDA_CACHE_PATH="$PRETRAIN_ROOT/cache/cuda"
export UV_CACHE_DIR="$PRETRAIN_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$PRETRAIN_ROOT/envs/python"
export TMPDIR="$PRETRAIN_ROOT/tmp"
export XDG_CACHE_HOME="$PRETRAIN_ROOT/cache"
export PYTHONNOUSERSITE=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_ETAG_TIMEOUT=30
export MODELSCOPE_CACHE="$PRETRAIN_ROOT/cache/modelscope"
export MODELSCOPE_HUB_CACHE="$PRETRAIN_ROOT/cache/modelscope-hub"
export MODELSCOPE_HOME="$PRETRAIN_ROOT/cache/modelscope"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export http_proxy=http://127.0.0.1:17890
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$http_proxy"
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy="$NO_PROXY"
mkdir -p "$PRETRAIN_ROOT"/{envs,cache,tmp,data,runs,exports,logs,tools}
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
