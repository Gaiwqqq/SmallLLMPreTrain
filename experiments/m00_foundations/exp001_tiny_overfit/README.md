# M0：Tiny Corpus Overfit

<!-- BEGINNER_GUIDE -->

## 初学者导读：先让模型背下来，检查学习链路

**这是保留的历史报告。** 本次远程结果请读 [运行记录](../../../docs/results.md)，术语见 [术语表](../../../docs/glossary.md)。

**为什么做？** 整个训练系统包含编码、输入、标签、loss、反向传播和保存。若最小数据都学不会，先修链路比扩大训练更有价值。

**为什么这样设计？** 只有 16 行语料，反复读一个固定 batch，不增加陌生文本。这样把泛化难度降到很低，主要检查模型能不能记忆。使用现成 Tokenizer 也只是减少编码变量，没有加载其模型权重。

**怎样判断？** 同时看 loss、下一 token 准确率和保存后重载。只看一条生成可能碰巧对；只看训练内存里的模型，无法证明保存文件可用。

**实际结论。** 第 223 次更新，loss 约 0.049，next-token accuracy=100%，重载后还能还原训练内容，最小链路通过。

**不能推出什么？** 模型从未经历真正陌生问题，能背训练文字不是能聊天。下一步换成真实文章，并单独留出验证文档。

下面保留原始设置和结果，想复核命令时再展开。

<details>
<summary>查看原始完整记录：参数、复现命令、指标与生成样例</summary>

## 目的

验证 Tokenizer、定长 packing、causal LM loss、反向传播、AdamW、
TensorBoard 和 checkpoint 保存/加载能够连通。这个实验只检查模型能否记住
固定 batch，不衡量泛化能力。

## 运行

```bash
conda activate dummym
python scripts/train/tiny_overfit.py \
  --tokenizer /path/to/Mistral-7B-v0.1/tokenizer.json \
  --device cuda
```

关键设置：16 行语料、Mistral 32K Tokenizer、每篇文档追加 EOS、不添加 BOS、
`4 × 128` 固定 batch、2,136,384 参数、AdamW、恒定学习率 `1e-3`、seed 2026。

## 结果

| 指标 | 结果 |
| --- | ---: |
| 初始 loss | 10.381273 |
| 最终 loss | 0.049335 |
| checkpoint 重新加载后的 loss | 0.047793 |
| next-token accuracy | 1.0000 |
| 完成步数 | 223 |

checkpoint 严格加载时所有参数名称均匹配。用训练语料前缀
`The small language model` 做 greedy decoding，模型生成：

```text
learns to predict the next token from the tokens that came before it.</s>
```

测试命令：

```bash
python scripts/inference/tiny_checkpoint_demo.py --device cuda
```

TensorBoard：

```bash
tensorboard --logdir runs/m00_tiny_overfit/tensorboard
```

## 结论

**通过。** 当前最小训练链路能够完全拟合固定训练数据，保存后的 checkpoint
也能恢复并复现训练内容。这个结果不能说明模型具备通用语言能力。

下一步是确定正式的 BOS/EOS 约定，并验证从 checkpoint 继续训练是否正常。

相关文件：

- 语料：`data/tiny_corpus.txt`
- 训练：`scripts/train/tiny_overfit.py`
- checkpoint 测试：`scripts/inference/tiny_checkpoint_demo.py`
- 本地产物：`runs/m00_tiny_overfit/`

</details>
