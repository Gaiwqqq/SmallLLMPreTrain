# 原生模型：从 token 到下一词预测

这个目录实现模型的数学计算。输入是一串 token ID，输出是每个位置对下一 token 的预测分数（logits）。训练 loss 衡量真实下一 token 是否获得较高概率。先理解下面的数据流，再读每个模块：

```text
token embedding
  -> N x (RMSNorm -> GQA causal SDPA -> residual
           -> RMSNorm -> SwiGLU MLP -> residual)
  -> RMSNorm
  -> bias-free LM head
  -> vocabulary logits
```

RoPE 让注意力感知位置；GQA 让多组 query 共用较少的 key/value 头，以减少相关存储和计算。RMSNorm 调整数值尺度，SwiGLU 负责非线性变换，residual 保留信息与梯度通道。Embedding 与输出层共享权重可减少参数。

这些选择来自常见 Llama 类结构，方便接入成熟工具；本项目没有逐项消融来证明每一种设计都最好。详细动机与张量形状见 [模型阶段](../../../../docs/01_foundations.md)。

## 最小例子

```python
import torch

from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM

config = MiniLlamaConfig(
    vocab_size=32_000,
    hidden_size=256,
    intermediate_size=688,
    num_hidden_layers=4,
    num_attention_heads=8,
    num_key_value_heads=2,
    max_position_embeddings=2_048,
)
model = MiniLlamaForCausalLM(config)
input_ids = torch.randint(0, config.vocab_size, (2, 128))
output = model(input_ids, labels=input_ids)
output.loss.backward()
```

例子中的 `(2, 128)` 表示两条、每条 128 token 的序列。`labels=input_ids` 时模型内部将预测与下一位置的标签对齐；`backward()` 计算梯度，还需要 optimizer.step() 才会更新参数。

原生模型用于理解计算，不包含 KV cache 或高性能服务循环。HF 转换在 checkpoint 模块，DDP 在 training 模块，聊天推理在 inference 模块；这样可以分别检查模型数学、分布式更新与部署格式。
