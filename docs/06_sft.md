# 聊天后训练

## 本阶段学什么

从续写模型学习 assistant 回复。

## 输入与阶段成果

Hugging Face 导出、SFT checkpoint、聊天模板与对照结果。

## 设计与可执行步骤

预训练学习文本分布；SFT 用对话格式教模型回答用户。小模型使用短对话，选择 HuggingFaceTB/smol-smoltalk。首轮全参数微调而不是 LoRA，避免适配器成为额外教学概念。

```bash
dummym-export --checkpoint ../runs/m04_base/checkpoint.pt \
  --tokenizer ../data/tokenizer/english32k --output ../exports/base
dummym-download --repo HuggingFaceTB/smol-smoltalk --files 0 --output ../data/raw/sft
python -m torch.distributed.run --standalone --nproc_per_node=4 --no-python \
  "$PRETRAIN_ROOT/envs/train/bin/dummym-sft" --model ../exports/base \
  --data-dir ../data/raw/sft --output ../runs/m09_sft --learning-rate 3e-5
```

导出对照 FP32 logits 和 loss 后才写模型文件。Chat template 使用 <|im_start|> / <|im_end|>，assistant 内容由 generation 标签圈定，user/system/padding 不监督。首轮不跨会话 packing。

全参数 SFT 保留 FP32 参数和 AdamW 状态，由 Trainer 在前向使用 BF16，以避免小学习率更新被 BF16 参数量化吞掉。预处理会过滤截断到 2048 后没有任何 assistant 预测目标的会话，防止全忽略标签导致无效 loss。

LR=1e-5/3e-5/6e-5/1e-4 的独立单卡短跑后，按开发集选定四卡正式配置，最多两轮。训练与验证按整个会话划分、精确去重；最终测试不参与选择。监测 SFT loss 同时检查回复质量，不能只凭最低 loss 选择聊天助手。

## 本次结果

待本次运行验证；历史实验结果不视为本次结果。实测状态见 [运行记录](results.md)。

## 下一步

按根 README 的学习顺序进入下一阶段；未通过验收先定位原因。
