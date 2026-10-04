# 分布式与性能

## 本阶段学什么

相同全局 batch，验证同步与恢复。

## 输入与阶段成果

单/双/四卡对照、尾部测试、恢复报告、吞吐和显存表。

## 设计与可执行步骤

先读 training/distributed.py，再读 recipe.py 和 engine.py。全局 batch=world_size × micro_batch × accumulation；例：4 卡 × 每卡 8 条 × 累积 8 次=256 条。

DDP 会平均各 rank 的梯度。局部 micro-batch loss 需乘 world_size × local_count/global_count。尾部不足时仍保持相同 collective 顺序；没有样本的 rank 用零权重 dummy forward，防止挂起，不计入训练 token。

```bash
pytest tests/unit/training -q
pytest tests/integration -q
python -m torch.distributed.run --standalone --nproc_per_node=4 --no-python \
  "$PRETRAIN_ROOT/envs/train/bin/dummym-pretrain" \
  --model-config configs/model/p099m.yaml --data-dir ../data/tokenized/reference100m \
  --output-dir ../runs/m03_ddp --total-tokens 99999744 \
  --global-batch-size 16 --micro-batch-size 4 --learning-rate 1e-3
```

验证在原模型上独立前向，最后聚合指标，避免不同 rank 验证条数不同导致 DDP forward collective 不匹配。checkpoint 保存各 rank RNG 和全局游标；恢复要求相同 world size、数据内容和 recipe。

吞吐按所有 rank 的实际 token 和同步墙钟时间计算。比较 micro-batch=4/8/16/32/64；global batch 固定为 256。torch.compile 正确性通过且稳态吞吐至少提高 10% 才启用。小模型单卡可放下，DDP 是正式路线；FSDP2 对照用于理解分片的通信与显存代价。

## 本次结果

待本次运行验证；历史实验结果不视为本次结果。实测状态见 [运行记录](results.md)。

## 下一步

按根 README 的学习顺序进入下一阶段；未通过验收先定位原因。
