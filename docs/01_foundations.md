# 模型原理：文字怎样变成一次参数更新？

这一阶段先回答一个小问题：给模型一段文字，它怎样得到“下一 token”的预测，并据此学习？先读 [阅读路线](reading_guide.md)；术语查 [术语表](glossary.md)。

## 用一个例子串起训练

假设一段文字编码成 `[A,B,C]`。模型看到 A 时预测 B，看到 A、B 时预测 C。它不能偷看 C 后再预测 C。因此我们使用 **causal attention**：每个位置只关注当前和过去的位置。

1. Tokenizer 把文字变成整数 ID，组成 `[B,T]` 的输入。
2. embedding 将每个 ID 变成 D 个数字，得到 `[B,T,D]`。
3. Transformer 的多层 attention 与 MLP 逐步处理这些表示。
4. 输出层给每个位置的 V 个候选 token 打分，得到 `[B,T,V]` 的 logits。
5. 交叉熵 loss 衡量正确的下一 token 获得了多大概率；概率越小，惩罚通常越大。
6. 反向传播计算梯度，AdamW 用梯度更新参数。

代码调用 `model(input_ids=ids, labels=ids)` 时，模型内部已经把预测与答案错开一位。数据端不要再 shift 一次，否则会让模型预测错误的位置。

## 为什么使用这些模型部件？

| 部件 | 先这样理解 | 本轮采用的原因与边界 |
|---|---|---|
| RMSNorm | 调整表示的尺度 | 减少层间数值尺度漂移；不是保证训练稳定的万能开关 |
| RoPE | 给 attention 提供位置信息 | 区分词语的位置；不表示当前模型已验证更长上下文 |
| GQA | 多个 query head 共享较少 K/V head | 减少 K/V 投影参数及后续推理中的 K/V 存储；共享也限制了表示方式 |
| SwiGLU MLP | 对表示做带门控的变换 | 提供 attention 之外的表达能力，采用 Llama-like 结构便于标准导出 |
| residual | 把输入加回变换结果 | 让信息与梯度能跨层传递 |
| 共享 embedding 与输出权重 | 输入识字表与输出打分表共享数字 | 小模型节省参数；本轮没有做不共享的消融对照 |

这些是架构选择的理由，不是本项目已经通过实验证明每个部件都优于替代方案。我们先选定一套结构，避免初期同时研究太多变量。

## 第一个实验为什么故意让模型“背答案”？

M0 反复训练同一个很小的固定 batch。真实泛化训练不能这样验收，但基础调试非常适合：若模型连少量文本都记不住，应先检查标签、mask、loss、梯度和学习率。

设计：固定语料、随机初始化模型、固定 seed、固定学习率，观察 next-token loss 和准确率；保存后重新加载，检查训练内容还能否恢复。这里使用现成 Mistral Tokenizer，只借用编码方式，没有借用其模型权重。自训 Tokenizer 放到后面的阶段，先减少变量。

本次 M0 在第 223 次更新达到 loss < 0.05，next-token accuracy 为 100%。**结论是最小学习链路可用。** 不能推出模型会回答陌生问题；这些答案来自它已经背过的固定内容。

## 动手和源码阅读

在远程源码目录激活环境后，可运行下面的教学实验。当前任务已有产物，重跑时使用新目录。

```bash
pytest tests/unit/model -q
python scripts/train/tiny_overfit.py \
  --tokenizer "$PRETRAIN_ROOT/data/reference_tokenizer/tokenizer.json" \
  --device cuda --output-dir "$PRETRAIN_ROOT/runs/m00_new_overfit"
```

源码按 `config.py → norm.py → rope.py → attention.py → mlp.py → model.py` 阅读，均位于 `src/dummym/models/llama_like/`。先在纸上标出每一步的 `[B,T,D]` 形状，再读 `scripts/train/tiny_overfit.py` 的一次更新。

教学 BPE 在 `src/dummym/tokenizer/bpe.py`：数相邻片段出现次数，合并最常见的一对，重复多次。正式大语料使用 Rust Tokenizers，以免教学实现成为计算瓶颈。

阶段成果是模型数学测试、tiny overfit 和保存/加载证据。下一阶段换成未见过的验证文档，检查模型是否只会背训练内容。详细证据见 [运行记录](results.md) 与 [M0 历史报告](../experiments/m00_foundations/exp001_tiny_overfit/README.md)。
