# 正确性实验：换文件格式是否改变模型

## 问题与原因

自写模型方便理解，标准 Llama 文件便于 TRL/vLLM 使用。权重名映射错误时，文件可能仍能加载，因此要直接比较数学结果。

## 为什么这样设计

固定同一权重和输入，用 FP32 对照映射前后的 logits、loss；单元测试也对照梯度。先严格加载所有参数，再保存标准 safetensors 和完整 Tokenizer 封装。

改变的是实现/存储接口，不是训练权重、词表或预测目标。共享 embedding 也必须保持同一约定。

## 结果与结论

对照通过，213M pilot 已实际导出并生成四个续写。导出成功证明受测转换一致、加载和生成可用。样例重复明显，不能把这一工程结论变成聊天质量结论。

下一步正式 Base 训练完成后做同样检查，再交给 SFT。证据：`tests/unit/model/test_huggingface_export.py`、`exports/pilot-base-lr3e4/export.json`、`runs/pilot-completions.json`；详见 [正式预训练](../../../docs/05_pretraining.md)。
