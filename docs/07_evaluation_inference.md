# 评测与推理

## 本阶段学什么

从可加载走向可交流。

## 输入与阶段成果

120 项最终测试报告、CLI/API、推理延迟与模型限制。

## 设计与可执行步骤

先冻结 40 项开发题与 120 项最终题。最终题分日常对话、常识、指令遵循、改写、摘要、三轮上下文六类，每类 20 项。开发集用于选配置，测试集只用于最终验收。

```bash
dummym-evaluate --model ../runs/m09_sft/final \
  --suite evaluation/chat_test.jsonl --output ../runs/evaluation/test.jsonl
dummym-chat --model ../runs/m09_sft/final
bash scripts/inference/serve.sh ../runs/m09_sft/final
```

目标：总通过率 >=70%，每类 >=60%；至少 90% 没有空回答、循环复读或角色泄漏。评分表必须由人工按 rubric 审阅，自动结构检查不能替代语义评分。摘要检查主要事实，改写检查原意，多轮检查上下文。

CLI 使用 /reset 和 /quit；上下文超限提示清空历史，不静默截断。vLLM 单卡部署，回环 8000 端口通过 SSH 访问，tensor parallel=1，避免小模型跨卡通信。

环境独立放到 envs/inference。安装 vLLM 前检查 PyTorch/CUDA 与驱动兼容；训练环境不升级。增加 lm-evaluation-harness 英文任务作为辅助评测，指标不等同于对话能力。

模型卡记录来源、许可、训练 token、架构、结果、知识错误和失败案例。不满足验收时明确写未达标。

## 本次结果

待本次运行验证；历史实验结果不视为本次结果。实测状态见 [运行记录](results.md)。

## 下一步

按根 README 的学习顺序进入下一阶段；未通过验收先定位原因。
