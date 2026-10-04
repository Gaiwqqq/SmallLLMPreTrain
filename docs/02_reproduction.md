# 基线复现

## 本阶段学什么

用相同预算对照历史实验。

## 输入与阶段成果

39M/99M 的 loss 曲线、恢复结果、实际吞吐和差异说明。

## 设计与可执行步骤

先下载 FineWeb-Edu 分片，将原 Mistral tokenizer.json 放到本项目 data/reference_tokenizer/，不得使用预训练模型权重。

```bash
python scripts/data/prepare_fineweb.py \
  --input-dir ../data/raw/fineweb --tokenizer ../data/reference_tokenizer/tokenizer.json \
  --output-dir ../data/tokenized/reference100m
python scripts/train/pretrain.py \
  --data-dir ../data/tokenized/reference100m --device cuda \
  --output-dir ../runs/m01_39m --stop-after-steps 100
python scripts/train/pretrain.py \
  --data-dir ../data/tokenized/reference100m --device cuda \
  --resume ../runs/m01_39m/checkpoint.pt
```

99M 使用 configs/model/p099m.yaml。原项目选定 LR=1e-3、warmup=300；复核 6e-4 与 1e-3 的两个 seed，再固定 LR 对比 warmup=100/300。独立实验各占一张卡，不互相抢显存。

本次原始分片可能与历史分片不同，验证 loss 不要求完全相同。记录数据指纹、版本与 seed 后再解释差异。旧报告位于 experiments/m01_pretraining、m02_recipe。

## 本次结果

39M 与六组 99M 均完成约 100M token 训练。39M 验证 loss 为 4.219103；99M 最好一组为 3.647846。两组 seed 均支持 LR=1e-3、warmup=300；详细表格和曲线见运行记录。 实测证据见 [运行记录](results.md)。

## 下一步

按根 README 的学习顺序进入下一阶段；未通过验收先定位原因。
