#!/usr/bin/env bash
# 两轮有界正式训练；冻结 test 只用于最终模型，不用于开发选优。
set -euo pipefail
cd /diff/gaiwq/llm_pretrain/repo
source scripts/remote/environment.sh
exec 9>"$PRETRAIN_ROOT/formal-curriculum-v2.lock"
flock -n 9
RUN="$PRETRAIN_ROOT/runs/sft_curriculum_v2_formal"
RAW="$PRETRAIN_ROOT/data/curriculum-v2"
READY="$PRETRAIN_ROOT/data/tokenized/curriculum-v2"
CONTROL="$PRETRAIN_ROOT/runs/curriculum-v2-formal-controller.json"
status() {
  python - "$CONTROL" "$1" <<'PY'
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]); temp=p.with_suffix('.json.tmp')
temp.write_text(json.dumps({'stage':sys.argv[2],'updated_at':time.time(),
                           'general_chat_accepted':False},indent=2)+'\n')
temp.replace(p)
PY
}
trap 'status failed' ERR
if [ -e "$RUN" ]; then
  echo 'Existing formal run: inspect checkpoints before retry; no overwrite.'
  exit 1
fi
status checking_frozen_test
python - <<'PY'
import hashlib,json
from pathlib import Path
p=Path('evaluation/chat_test_v2.jsonl')
m=json.loads(Path('evaluation/chat_test_v2.manifest.json').read_text())
assert hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256']
assert len(p.read_text().splitlines())==m['cases']==120
PY
status checking_prepared_data
dummym-sft --model "$PRETRAIN_ROOT/exports/chat" --data-dir "$RAW" \
  --prepared-data "$READY" --output "$RUN" --prepare-only
status training
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --nnodes 1 --nproc_per_node 4 --master_port 29644 --no-python \
  "$PRETRAIN_ROOT/envs/train/bin/dummym-sft" \
  --model "$PRETRAIN_ROOT/exports/chat" --data-dir "$RAW" --prepared-data "$READY" \
  --output "$RUN" --learning-rate 3e-5 --epochs 2 --batch-size 4
status generating_final_acceptance
CUDA_VISIBLE_DEVICES=1 dummym-evaluate --model "$RUN/final" \
  --suite "$PRETRAIN_ROOT/repo/evaluation/chat_test_v2.jsonl" \
  --output "$RUN/fresh-chat-test.jsonl"
CUDA_VISIBLE_DEVICES=2 dummym-evaluate --model "$RUN/final" \
  --suite "$PRETRAIN_ROOT/repo/evaluation/chat_dev.jsonl" \
  --output "$RUN/general-dev.jsonl"
status exporting_and_uploading
mkdir -p "$PRETRAIN_ROOT/exports/curriculum-v2-formal"
cp -a "$RUN/final/." "$PRETRAIN_ROOT/exports/curriculum-v2-formal/"
printf '%s\n' '# Curriculum v2 formal SFT' '' \
  'Two epochs from original Chat, LR 3e-5. Final semantic acceptance pending.' \
  > "$PRETRAIN_ROOT/exports/curriculum-v2-formal/README.md"
dummym-publish --root "$PRETRAIN_ROOT" --folder "$PRETRAIN_ROOT/exports/curriculum-v2-formal" --docs
status awaiting_final_semantic_review
