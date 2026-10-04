# 正式预训练

## 本阶段学什么

把正确性流程扩展到约 213M 参数。

## 输入与阶段成果

Base checkpoint、曲线、独立验证和模型卡。

## 设计与可执行步骤

模型参数见 configs/model/p213m.yaml：16 层，D=1024，MLP=2816，Q/KV heads=16/4，2048 上下文，共享 embedding。自训 Tokenizer 后重新随机初始化，不能沿用原 Mistral 的 embedding。

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 --no-python \
  "$PRETRAIN_ROOT/envs/train/bin/dummym-pretrain" \
  --model-config configs/model/p213m.yaml --data-dir ../data/tokenized/english \
  --output-dir ../runs/m04_base --total-tokens 9999996928 \
  --global-batch-size 256 --micro-batch-size 16 --learning-rate 6e-4 \
  --max-hours 96
```

上述 token 数需以数据 manifest 为准，不得大于实际完整序列数。正式规模用一亿 token 对照 LR=3e-4/6e-4/1e-3，选择稳定且验证 loss 最低的一档。AdamW betas=(0.9,0.95)，weight decay=0.1，裁剪=1.0。warmup 占更新数 2%，cosine 衰减到峰值的 10%。

实测稳定吞吐 × 96 小时 × 0.85 为可用预算，向下取整到一亿 token，上限 20B。数据不足先补数据。运行前冻结 budget 和 schedule，暂停不改变计划。

每 500 更新保存恢复点，10%/25%/50%/完成或暂停时生成不可变 milestone，ready.json 最后写入。独立上传进程只处理 ready 产物，上传失败不影响训练。

验收：有限 loss/梯度，恢复通过，验证有改善，权重严格加载，续写可运行。能力改善需由正式评测证明。

## 本次结果

213M 的四个学习率/随机种子 pilot 已并行启动，每卡一个任务；完整数据准备同时进行。正式 Base 训练尚未开始，token 预算尚未冻结。 实测证据见 [运行记录](results.md)。

## 下一步

按根 README 的学习顺序进入下一阶段；未通过验收先定位原因。
