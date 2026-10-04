# 数据与 Tokenizer

## 本阶段学什么

先隔离文档，再编码成训练序列。

## 输入与阶段成果

原始文件 manifest、去重统计、自训 32K BPE、训练/验证/测试二进制分片。

## 设计与可执行步骤

预训练按实际 token 混合：FineWeb-Edu 70%、Cosmopedia-v2 25%、TinyStories 5%。前者提供通用英文，后两者补充解释性文字和简单连贯故事；比例是本轮实验假设。

```bash
dummym-download --repo HuggingFaceFW/fineweb-edu --prefix data/ \
  --files 24 --output ../data/raw/fineweb
dummym-download --repo HuggingFaceTB/smollm-corpus --prefix cosmopedia-v2/ \
  --files 16 --output ../data/raw/cosmopedia
dummym-download --repo roneneldan/TinyStories --prefix data/train- \
  --files 0 --output ../data/raw/tinystories
dummym-clean --source fineweb=../data/raw/fineweb \
  --source cosmopedia=../data/raw/cosmopedia --source tinystories=../data/raw/tinystories \
  --output ../data/clean/english
dummym-tokenizer --input ../data/clean/english/fineweb.train.jsonl \
  ../data/clean/english/cosmopedia.train.jsonl ../data/clean/english/tinystories.train.jsonl \
  --output ../data/tokenizer/english32k
dummym-pack --clean ../data/clean/english \
  --tokenizer ../data/tokenizer/english32k/tokenizer.json \
  --train-tokens 10000000000 --output ../data/tokenized/english10b
```

下载记录不可变 revision 和 SHA-256。正文轻量规范化后精确去重，并用采样五词 shingle 的 MinHashLSH 做近重复过滤；它有误检和漏检，README 必须说明。不同来源共享去重范围，近重复过滤发生在 split 之前。

通过正文哈希划分文档，train/validation/test 约 99/0.5/0.5%。BPE 只看训练文档。预留 BOS/EOS/PAD 和聊天 token，编码追加 EOS，分别 packing 成 2048 长度的 uint16 序列。模型内部执行 label shift。

验收：split 无精确重叠；词表 32K；特殊 token ID 正确；round-trip、ID 范围、尺寸、来源配额与指纹通过。语料不足时报错，绝不悄悄重复文档凑预算。

## 本次结果

待本次运行验证；历史实验结果不视为本次结果。实测状态见 [运行记录](results.md)。

## 下一步

按根 README 的学习顺序进入下一阶段；未通过验收先定位原因。
