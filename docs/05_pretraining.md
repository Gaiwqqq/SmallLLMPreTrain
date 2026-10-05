# 正式预训练：让随机权重从大量文章中学习

前面的实验已经确认：数据可读、训练会更新参数、四卡同步正确、断点可恢复。正式训练现在要回答另一件事：更多唯一文本能否改善模型对陌生文章的预测与续写？

## 为什么选约 213M 参数？

这是一轮学习与交付兼顾的取舍：比 39M/99M 有更多表达容量，仍容易放入单张 H20，从而先用 DDP 扩大吞吐，不把模型分片作为入门前提。我们没有做等计算预算的模型规模对照，不能说 213M 是最优规模。

结构为 16 层、表示维度 1024、MLP 2816、16 个 query head/4 个 K/V head、上下文 2048、共享 embedding。每个数字的具体含义见 [模型原理](01_foundations.md) 和 `configs/model/p213m.yaml`。

Tokenizer 自训后，模型重新随机初始化。现成 PyTorch/Transformers 负责执行和导出，不提供任何预训练模型权重。

## 为什么 99M 的 LR 不能直接搬过来？

213M 的参数规模、Tokenizer、语料和 global batch 都变了。旧配置好，不代表新配置也好。因此先做四个约 100M-token pilot（先导短实验）：

| LR | seed | 验证 loss |
|---|---:|---:|
| 3e-4 | 2026 | 5.988814 |
| 6e-4 | 2026 | 6.031944 |
| 1e-3 | 2026 | 6.136201 |
| 6e-4 | 2027 | 6.013090 |

同一 seed 的前三组主要改变 LR；第四组对 6e-4 做一次随机性复核。每组读同一份 99,997,696-token 混合数据，使用同一验证集。本轮选择 **3e-4**。

这只是短预算下的配置筛选，不证明长预算的全局最优，也没有给三个 LR 各做多 seed 的完整统计。决策记录保存在 `runs/lr-selection.json`。

## 先导实验为什么有价值，即使样例很差？

pilot 能导出、能续写，但四个样例重复明显。例如 “Water is important because” 后反复出现 “the little girl”。它说明小预算模型还没有达到沟通目标，也提醒我们不能只读 loss 曲线。

100M token 对 213M 参数来说，每参数约对应 0.47 个训练输入 token；正式 7.7B 约对应 36 个。这是规模比值，不是质量保证或最优训练定律。扩大唯一语料是下一步实验，改善是否真实发生仍需验证与生成检查。

## 正式预算怎样确定？

同时受三个上限约束：计划最多 20B token、按稳定吞吐和最多 96h 估算的计算预算、保持 70/25/5 配比时的唯一数据容量。最后一个上限在本轮最紧，原因见 [数据阶段](03_data_tokenizer.md)。

提案为 7.7B，packing 对齐后实际输入 **7,699,996,672 token**；验证/测试各 9,496,576 token。预算、LR 和 schedule 在开始前冻结，不能因为中途暂停就重算剩余学习率曲线。

global batch=256，每卡 micro-batch=16，四卡累积 4 次。AdamW betas=(0.9,0.95)，weight decay=0.1，梯度裁剪=1。warmup 占总更新数 2%，之后 cosine 衰减到峰值的 10%。

weight decay 给参数增长施加约束；梯度裁剪限制一次异常大梯度；它们都不保证事实正确。BF16 用于计算，FP32 保存参数与优化器状态，降低细小更新被量化吞掉的风险。

## 怎样观察这次长训练？

正式训练已启动，模型在学习过程中。训练更新吞吐约 21.8 万输入 token/s；不能用这个数字直接保证总完成时间，因为还有验证与保存。

```bash
# 只查看，不启动第二份训练。
tail -n 3 "$PRETRAIN_ROOT/runs/m04_base/metrics.jsonl"
cat "$PRETRAIN_ROOT/runs/pretraining-proposal.json"
```

每 500 次更新验证/保存，关键进度 10%/25%/50%/结束或暂停保留不可变 milestone。`ready.json` 在完整保存后写入，独立上传器归档到 ModelScope。保存完整状态的理由是能继续同一个实验，而不只是用权重生成文本。

复现命令如下，仅在新实验且相同数据已准备好时使用；输出目录须换新。不要在当前四卡训练旁再启动一份：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 --no-python \
  "$PRETRAIN_ROOT/envs/train/bin/dummym-pretrain" \
  --model-config configs/model/p213m.yaml \
  --data-dir "$PRETRAIN_ROOT/data/tokenized/english" \
  --output-dir "$PRETRAIN_ROOT/runs/m04_new_base" --total-tokens 7699996672 \
  --global-batch-size 256 --micro-batch-size 16 --learning-rate 3e-4 \
  --max-hours 96
```

## 结束时怎样得出结论？

先看实现：loss/梯度有限，完整预算或明确暂停点，恢复和加载可用。再看学习：固定验证集是否改善。最后看能力：统一续写样例和独立评测是否改善，是否仍重复、跑题或编造。

Base 仍是文章续写模型，尚未被教会 user/assistant 的回复规则。阶段产物为 Base 权重、Tokenizer、曲线、模型卡和完整恢复点；下一阶段用 [SFT](06_sft.md) 学习对话回复。最新完成状态以 [自动运行记录](results.md) 为准。
