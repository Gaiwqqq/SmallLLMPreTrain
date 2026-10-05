# 冻结的英文交流题集

`chat_dev.jsonl` 有 40 项开发题；`chat_test.jsonl` 有 120 项最终题，六类各 20 项。
题目在正式训练前冻结，开发题用于选配置，最终题不用于调参。每项有固定 ID、
用户轮次和评分 rubric。多轮题连续生成真实模型回复，不用预写的 assistant 答案。

运行 `dummym-evaluate` 生成回答后，逐项审阅，创建 JSONL 评分：

```json
{"id": "knowledge-01", "pass": true, "reason": "回答 Paris，且没有相反陈述"}
```

每项必须有唯一评分、布尔 pass 和基于实际回答的理由。汇总命令：

```bash
dummym-evaluate --predictions ../runs/evaluation/test.jsonl \
  --reviews ../runs/evaluation/test-reviews.jsonl --output ../runs/evaluation/report.json
```

目标：总通过率 70%、各类 60%、结构正常回答 90%。自动检查空回答、循环复读、
角色泄漏和上下文超限；这些检查不能代替语义验收。题集刻意覆盖基础能力，
不能据此宣称与成熟通用助手能力相当。最终报告同时公开失败例子。

## 初学者如何理解评分

训练 validation loss 问的是“预测真实下一 token 有多准”；这里的 chat_dev/chat_test 问的是“回答是否满足用户要求”。它们用途不同，loss 下降不能替代逐题读答案。

例如用户要求“用一句话说明水为何重要”，只说一遍合理原因可通过；反复说 “the little girl” 即使没有程序报错也应失败。评分理由必须引用实际表现，不能按模型大小或期望评分。尚未审阅的语义结果应保持未评分，不把结构检查通过当作语义通过。

70%/60%/90% 是本项目预先设定的验收目标，不是行业标准。最终题看过之后再针对失败例子调参，会影响测试独立性；后续改进须明确披露，并使用新的独立题集。
