# SFT 后的推理恢复：训练完成不等于服务可用

本页记录依赖兼容故障的修复方案。**当前修复在本地已准备、校验，尚未应用训练机**：本次执行环境在创建 SSH socket 时返回 `PermissionError: Operation not permitted`，不能读取远程日志或验证部署结果。不要把下面的计划写成已完成结果。

## 已知事实与待核验项

阶段通知报告：正式 SFT 完成 7060 步，eval_loss 约 1.149；随后 vLLM 离线测试报 `TokenizersBackend has no attribute all_special_tokens_extended`。这些信息来自通知，待重新连接后核对 `runs/m09_sft/summary.json`、`workflow.json` 与 `logs/vllm-offline-smoke.log`。

SFT loss 衡量示范回复的下一 token 预测；服务异常发生在加载接口。二者是不同的环节，不需要因此重新训练权重。最终能否交流还要检查真实回答。

## 为什么固定这些版本？

[vLLM 0.11.0 的 Tokenizer 源码](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/transformers_utils/tokenizer.py) 直接读取 `all_special_tokens_extended`；[它的依赖声明](https://github.com/vllm-project/vllm/blob/v0.11.0/requirements/common.txt) 只要求 Transformers >=4.55.2，没有主版本上限。旧 vLLM 与较新的 Tokenizer 接口发生漂移，是报告中异常的合理解释；实际安装版本必须现场读取。

`requirements/inference.txt` 固定 vLLM 0.11.0、Transformers 4.57.1、tokenizers 0.22.1。保留当前推理引擎版本，使用训练工程已采用的 Transformers 4 系列接口，减少其他依赖同时变化。这个候选组合仍须通过远程依赖解析、Tokenizer 检查和真实 GPU 推理后才能称为修复成功。

## 为什么需要三个检查？

1. **依赖与 Tokenizer 预检**：核对版本，调用 vLLM 的真实 Tokenizer 缓存包装，检查编码和特殊 ID 是否一致。在 GPU 分配前发现同类问题。
2. **离线推理**：真实加载 Chat 权重并生成非空回答，确认模型和计算引擎能协同执行。
3. **服务检查**：启动回环 8000 服务，检查 `/v1/models` 和 `/v1/chat/completions`。离线通过仍不能证明 HTTP 服务启动成功。

服务只接管具有本工程 PID/模型记录的进程；端口被其他服务占用时停止，避免把旧服务的健康响应误认为新模型通过。启动失败时清理本次创建的子进程。服务状态写入 `runs/vllm-service.json`，日志在 `logs/vllm-service.log`。

ChatML 回复结束标记是 ID 5，文章 EOS 是 ID 2。API 请求显式传入 `stop_token_ids: [2, 5]`，与项目 CLI/评测的停止规则保持一致。普通客户端也应带上这个参数。

## 恢复步骤

先同步本次改动的代码、requirements 与文档到远程 `repo/`，恢复远程 17890 到本机 7890 的反向代理。然后在训练机执行：

```bash
cd /diff/gaiwq/llm_pretrain/repo
nohup bash scripts/remote/recover_inference.sh \
  > /diff/gaiwq/llm_pretrain/logs/inference-recovery.log 2>&1 < /dev/null &
```

脚本检查代理、记录当前版本，再恢复 `week.py`。新加的 `pin-inference-tokenizer-v1` 阶段会执行，即使旧的 `install-vllm` 已标记 complete；已完成的预训练和 SFT 会被跳过。后续完成依赖校验、离线推理与服务检查，然后再次同步 Chat 导出、阶段文档和缺失的关键 checkpoint，保存推理环境版本清单。

若控制器失败，脚本立即退出并保留日志，不写成功状态。若上传失败，已启动的回环服务可继续存在；上传须重试并核对。

## 还剩什么才算达到目标？

读取冻结 120 题的实际回答，逐题按 rubric 给出通过与理由，再汇总总分、各类分数和结构正常率。未看到回答时不能生成“通过”评分。服务可用、loss 下降和语义合格需要各自的证据。

当前本地验证只覆盖依赖漂移拒绝、服务归属、占用端口保护和启动失败清理，以及源码/脚本静态检查。远程 GPU、HTTP、完整题集评分和 ModelScope 验证仍待连接恢复。
