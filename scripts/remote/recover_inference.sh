#!/usr/bin/env bash
# 在训练机源码目录执行；使用已完成阶段记录恢复，不重新训练。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/remote/environment.sh

# 防止在控制器仍运行时重复启动流程；发布器也有自己的独占锁。
if pgrep -u "$(id -u)" -f '[p]ython.*scripts/remote/week.py' >/dev/null; then
  echo 'Training controller already running; inspect workflow.json.' >&2
  exit 1
fi

# 下载严格使用本机 7890 的反向隧道；失败时不切换到直连。
"$PRETRAIN_ROOT/envs/train/bin/python" - <<'PYPROXY'
import urllib.request
proxy = urllib.request.ProxyHandler({'https': 'http://127.0.0.1:17890'})
try:
    with urllib.request.build_opener(proxy).open('https://pypi.org/simple/transformers/', timeout=15) as response:
        if response.status != 200:
            raise RuntimeError('Proxy did not return a successful registry response')
except Exception:
    raise RuntimeError('Restore reverse tunnel 127.0.0.1:17890 -> local 7890 before recovery') from None
print('Download proxy verified')
PYPROXY

# 输出依赖现状便于比对，不打印认证环境。
"$PRETRAIN_ROOT/envs/inference/bin/python" - <<'PY'
from importlib.metadata import version
for package in ('vllm', 'transformers', 'tokenizers'):
    print(f'{package}=={version(package)}')
PY

# 前台恢复控制器；应在 tmux 中或用 nohup 运行此脚本。
# 失败即退出，保留诊断日志；已有完成阶段会被跳过。
"$PRETRAIN_ROOT/envs/train/bin/python" scripts/remote/week.py

# 完成记录可能来自上一次运行，另核对服务当前是否存活。
"$PRETRAIN_ROOT/envs/train/bin/python" scripts/inference/start_service.py \
  --root "$PRETRAIN_ROOT" --model "$PRETRAIN_ROOT/exports/chat"

# 再次同步 Chat 导出及缺失恢复点，确保最终交付不依赖旧发布器存活。
"$PRETRAIN_ROOT/envs/train/bin/dummym-publish" --docs \
  --folder "$PRETRAIN_ROOT/exports/chat"
"$PRETRAIN_ROOT/tools/uv" pip freeze \
  --python "$PRETRAIN_ROOT/envs/inference/bin/python" \
  > "$PRETRAIN_ROOT/envs/inference.lock.txt"

echo 'Inference recovered. Review runs/evaluation/test.jsonl before semantic acceptance.'
