#!/usr/bin/env bash
# 四卡纠正实验：所有长任务由脱离 SSH 的会话运行，保留原 Chat 服务。
set -euo pipefail
cd /diff/gaiwq/llm_pretrain/repo
source scripts/remote/environment.sh
exec 9>"$PRETRAIN_ROOT/corrective-curriculum-v2.lock"
flock -n 9
RUN="$PRETRAIN_ROOT/runs/sft_curriculum_v2_corrective"
RAW="$PRETRAIN_ROOT/data/curriculum-v2"
READY="$PRETRAIN_ROOT/data/tokenized/curriculum-v2"
MODEL="$PRETRAIN_ROOT/exports/chat"
if [ -e "$RUN" ]; then
  echo 'Existing experiment directory: inspect before retry; no automatic overwrite.'
  exit 1
fi
python - <<'PYCODE'
import hashlib,json
from pathlib import Path
p=Path('evaluation/chat_test_v2.jsonl')
m=json.loads(Path('evaluation/chat_test_v2.manifest.json').read_text())
assert hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256']
assert len(p.read_text().splitlines())==m['cases']==120
PYCODE
python scripts/posttrain/build_curriculum_v2.py --output "$RAW" \
  --train-count 32000 --eval-count 600 \
  --mix-existing "$PRETRAIN_ROOT/data/tokenized/sft" --mix-count 128000
python - "$RAW/download.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); obj=json.loads(p.read_text())
obj['revision']='curriculum-v2-20261006'
obj['note']='Six-family corrective SFT; held-out synthetic diagnostics do not certify general chat'
p.write_text(json.dumps(obj,indent=2)+'\n')
PY
dummym-sft --model "$MODEL" --data-dir "$RAW" --prepared-data "$READY" \
  --output "$RUN" --prepare-only
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --nnodes 1 --nproc_per_node 4 --master_port 29643 --no-python \
  "$PRETRAIN_ROOT/envs/train/bin/dummym-sft" \
  --model "$MODEL" --data-dir "$RAW" --prepared-data "$READY" \
  --output "$RUN" --learning-rate 3e-5 --epochs 1 --batch-size 4
CUDA_VISIBLE_DEVICES=1 dummym-evaluate --model "$RUN/final" \
  --suite "$PRETRAIN_ROOT/repo/evaluation/chat_dev.jsonl" \
  --output "$RUN/general-dev.jsonl"
CUDA_VISIBLE_DEVICES=2 dummym-evaluate --model "$RUN/final" \
  --suite "$RAW/test.jsonl" --output "$RUN/fresh-diagnostic-test.jsonl"
# 原始回复生成完成后仍须逐题语义审阅；不会按结构 flags 判定通过。
mkdir -p "$PRETRAIN_ROOT/exports/curriculum-v2"
cp -a "$RUN/final/." "$PRETRAIN_ROOT/exports/curriculum-v2/"
printf '%s\n' '# Curriculum v2 corrective SFT' '' \
  'General chat acceptance pending. Do not interpret synthetic diagnostic scores as acceptance.' \
  > "$PRETRAIN_ROOT/exports/curriculum-v2/README.md"
dummym-publish --root "$PRETRAIN_ROOT" --folder "$PRETRAIN_ROOT/exports/curriculum-v2" --docs
echo 'Training/evaluation/export finished; general semantic review pending.'
