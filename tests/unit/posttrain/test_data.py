"""截断、监督边界、TRL 兼容性和不可变预处理产物的回归测试。"""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("trl")
from datasets import Dataset
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from dummym.posttrain.data import (
    data_contract,
    encode_conversation,
    has_targets,
    load_prepared,
    prepare_data,
)
from dummym.tokenizer.train import CHAT_TEMPLATE, SPECIAL_TOKENS


@pytest.fixture
def tokenizer():
    vocab = {
        word: index
        for index, word in enumerate(
            SPECIAL_TOKENS
            + [
                "user",
                "assistant",
                "Hello",
                "Good",
                "morning",
                "long",
            ]
        )
    }
    core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=core,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="</s>",
        chat_template=CHAT_TEMPLATE,
        additional_special_tokens=SPECIAL_TOKENS[4:],
    )


def conversation(prompt="Hello", reply="Good morning"):
    return {
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]
    }


def test_truncation_removes_unlearnable_prompts(tokenizer):
    row = encode_conversation(conversation("long " * 100), tokenizer, max_length=32)
    assert len(row["input_ids"]) == len(row["assistant_masks"]) == 32
    assert not has_targets(row)
    assert not has_targets({"assistant_masks": [1, 0, 0]})


def test_collator_masks_user_and_padding(tokenizer):
    rows = [
        encode_conversation(conversation(), tokenizer),
        encode_conversation(conversation(reply="Good morning " * 8), tokenizer),
    ]
    batch = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id)(rows)
    for index, row in enumerate(rows):
        for position, supervised in enumerate(row["assistant_masks"]):
            assert (batch["labels"][index, position].item() != -100) == bool(supervised)
        assert (batch["labels"][index, len(row["input_ids"]) :] == -100).all()


def test_encoding_matches_pinned_trl_preparation(tokenizer, tmp_path):
    rows = [conversation(), conversation(reply="Good morning " * 40)]
    args = SFTConfig(
        output_dir=str(tmp_path),
        use_cpu=True,
        report_to="none",
        max_length=32,
        assistant_only_loss=True,
        bf16=False,
    )
    reference = SFTTrainer._prepare_dataset(
        SimpleNamespace(_is_vlm=False),
        Dataset.from_list(rows),
        tokenizer,
        args,
        False,
        None,
        "reference",
    )
    for index, row in enumerate(rows):
        expected = encode_conversation(row, tokenizer, max_length=32)
        for key in ("input_ids", "assistant_masks"):
            assert expected[key] == reference[index][key]


def test_prepared_split_counts_contract_and_corruption(tokenizer, tmp_path):
    data, model, output = tmp_path / "data", tmp_path / "model", tmp_path / "prepared"
    (data / "data").mkdir(parents=True)
    model.mkdir()
    tokenizer.save_pretrained(model)
    rows = [conversation(f"Hello {i}") for i in range(10)]
    rows += [rows[0], conversation("long " * 3000)]
    pq.write_table(pa.Table.from_pylist(rows), data / "data/train-0.parquet")
    (data / "download.json").write_text(
        json.dumps(
            {"status": "complete", "revision": "test", "files": [{"path": "data/train-0.parquet"}]}
        )
    )
    prepare_data(data, model, output, tokenizer, workers=1)
    contract = data_contract(data, model, tokenizer)
    split = load_prepared(output, contract)
    assert len(split["train"]) == 9 and len(split["test"]) == 1
    assert all(has_targets(row) and len(row["input_ids"]) <= 2048 for row in split["train"])
    with pytest.raises(ValueError, match="mismatch"):
        load_prepared(output, {**contract, "max_length": 1024})
    arrow = next(output.rglob("*.arrow"))
    with arrow.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_prepared(output, contract)
