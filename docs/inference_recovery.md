# SFT 后的推理恢复：训练完成不等于服务可用

本次修复已实际部署。现场读取到旧推理环境为 vLLM 0.11.0、Transformers 5.18.0、tokenizers 0.23.2；固定至 Transformers 4.57.1、tokenizers 0.22.1 后，Tokenizer 预检、GPU 离线推理和真实 HTTP 请求均通过。

正式 SFT 已核验完成 7,060 步，验证 loss 为 1.149027。2026-10-06 本次续接通过 SSH 再次核对 `/v1/models`，服务加载 `exports/chat`，模型 ID 为 `dummym-english`。这证明接口恢复；首轮语义验收仍然未达标，见 [交付报告](first_round_delivery.md)。

## 为什么固定这些版本？

[vLLM 0.11.0 的 Tokenizer 源码](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/transformers_utils/tokenizer.py) 直接读取 `all_special_tokens_extended`；[它的依赖声明](https://github.com/vllm-project/vllm/blob/v0.11.0/requirements/common.txt) 只要求 Transformers >=4.55.2，没有主版本上限。旧 vLLM 与较新的 Tokenizer 接口发生漂移，与现场安装的 Transformers 5 系列接口不兼容，造成此次故障。

`requirements/inference.txt` 固定 vLLM 0.11.0、Transformers 4.57.1、tokenizers 0.22.1。保留当前推理引擎版本，使用训练工程已采用的 Transformers 4 系列接口，减少其他依赖同时变化。这个组合已通过远程依赖解析、Tokenizer 检查及真实 GPU 推理。

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

## 恢复后的能力结论

已逐题读取冻结 120 题，保存通过判定及理由：总通过率 38.3%，各类结果见交付报告。服务可用、loss 下降和语义合格需要各自证据。本次恢复解决了接口问题，下一轮课程实验继续处理指令与上下文不足。
