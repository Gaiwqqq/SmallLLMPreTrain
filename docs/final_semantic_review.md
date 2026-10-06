# 最终语义验收：读实际回答，再判断是否达到交流目标

## 本次验收已完成

已重新连接训练机，核对正式模型与全部评测记录，逐轮审阅 120 test +40 dev 项。最终 test 15/120（12.5%），accepted=false；开发集另计 9/40（22.5%）。全部判定与理由已保存，完整结果见 [最终交付](final_delivery.md)。下文保留可复核流程，不是待完成的工作。

## 为什么需要先核验文件？

题集在训练前冻结，六类各 20 题。`collect_final_review.py` 核对题集 SHA-256、题目 ID、用户提问、rubric 和生成侧记录的模型路径。模型必须是 `runs/sft_curriculum_v2_formal/final`，而不是仍在服务中的旧 `exports/chat`。随后按每个对话轮次整理完整回复。

这能避免读错模型、漏题或把已经改过的题目称作原始验收。它不判断事实正确性：输出中的 `accepted=None` 和待填写的 `pass=None` 明确表示尚未审阅，不是零分，也不是通过。

## 恢复连接后直接执行

先读取控制器及训练摘要，不重复启动训练。同步新的审阅脚本后，在训练机执行：

```bash
cd /diff/gaiwq/llm_pretrain/repo
source scripts/remote/environment.sh
python scripts/posttrain/collect_final_review.py \
  --output "$PRETRAIN_ROOT/runs/sft_curriculum_v2_formal/final-review-packet"
```

现有输出目录不覆盖，以保护可能已经填写的判定。若目录已存在，先读其 `packet.json` 和原始回复哈希。材料包含最终 120 题、开发 40 题、生成模型与题集元信息、完整逐轮对照 `review.md` 及空审阅模板。

逐题读取真实回答，填写 `id`、布尔 `pass`、与实际错误或正确内容对应的 `reason` 和 reviewer。不得根据非空、关键词或结构 flags 批量生成通过分数。最终 test 用于验收；40 题 dev 用于解释变化，两者分别汇总，不合并为 160 题的最终分数。

```bash
dummym-evaluate \
  --predictions "$PRETRAIN_ROOT/runs/sft_curriculum_v2_formal/fresh-chat-test.jsonl" \
  --reviews "$PRETRAIN_ROOT/runs/sft_curriculum_v2_formal/final-reviews.jsonl" \
  --output "$PRETRAIN_ROOT/runs/sft_curriculum_v2_formal/final-report.json"
```

汇总器拒绝缺题、重复 ID、未填写布尔判定或没有理由的审阅。总通过率至少 70%、六类分别至少 60%、结构正常率至少 90% 才判定 accepted。若不通过，保留真实结果与可恢复模型，交付明确的能力边界。

## 判定中容易误解的地方

- 指令题必须满足明确格式和数量；“差不多正确”不能满足 only/exactly。
- 知识题出现正确关键词，但伴随重大相关错误，仍不通过。
- 礼貌改写要保留原请求含义、物品、地点与时间；把请求改成自己的承诺或写无关长信不通过。
- 本次摘要明确要求一整句，原样复制多句不能通过；必须保留 rubric 指定的主要事实。
- 上下文题必须核对全部任务轮次；中间数量算错而末轮姓名答对不能掩盖错误。
- LLM 辅助审阅不是独立人工认证；报告保留方法、局限与边界题理由，不夸大结论。

## 如何收尾？

保存原始回复、逐题理由、汇总报告、评分方法和文件哈希，上传到同一私有 ModelScope 工程，同时同步最终模型 README 与教学文档。控制器状态只有在这些步骤完成后才记录语义验收结果；不要将“待审阅”提前改成“通过”。根据实际结果决定服务使用哪一份权重，保持报告与服务模型一致。

审阅脚本的静态检查与 120+40 项测试材料核验已通过；实际语义评分已在训练机汇总并复核。报告与模型卡归档完成后，控制器写入语义审阅完成和真实 accepted 值。
