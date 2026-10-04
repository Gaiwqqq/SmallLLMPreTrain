# 模型原理

## 本阶段学什么

从 token 到梯度更新。

## 输入与阶段成果

单元测试、tiny overfit、简化 BPE 合并规则。

## 设计与可执行步骤

阅读顺序：config.py → norm.py → rope.py → attention.py → mlp.py → model.py，再读 scripts/train/tiny_overfit.py。

输入 [B,T] 先进入 embedding [B,T,D]，每层执行 Pre-Norm attention 和 SwiGLU，最后输出 [B,T,V] logits。RMSNorm 稳定尺度；RoPE 编码位置；GQA 共享 K/V heads；SwiGLU 通过门控扩展表达能力；共享 embedding 减少参数量。

next-token loss：输入 `[A,B,C]` 的前两个 logits 分别预测 B 和 C，不能让 A 预测自己。未来 token 不得被 attention 看到。

```bash
pytest tests/unit/model -q
python scripts/train/tiny_overfit.py \
  --tokenizer ../data/reference_tokenizer/tokenizer.json \
  --device cuda --output-dir ../runs/m00_tiny_overfit
```

验收：loss < 0.1，next-token accuracy > 99%。过拟合成功只证明链路可用，不证明泛化或聊天能力。

src/dummym/tokenizer/bpe.py 用字符和频率演示相邻符号合并；正式 BPE 使用 Rust Tokenizers，教学版不处理大型语料。

## 本次结果

M0 在第 223 次更新达到 loss < 0.05；HF 导出的 logits、loss 和梯度对照通过。这证明实现可以学习和转换，但过拟合不证明泛化能力。 实测证据见 [运行记录](results.md)。

## 下一步

按根 README 的学习顺序进入下一阶段；未通过验收先定位原因。
