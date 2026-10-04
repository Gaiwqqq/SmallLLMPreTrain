"""新训练循环的尾部权重、恢复和数据契约回归测试。"""

from dataclasses import replace
import json

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models

from dummym.data.packed import load_packed
from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM
from dummym.training.distributed import ProcessContext
from dummym.training.engine import run, train_update
from dummym.training.recipe import TrainRecipe, make_optimizer
from dummym.utils.files import sha256_file


@pytest.fixture
def data(tmp_path):
    directory = tmp_path / "packed"
    directory.mkdir()
    tokenizer = Tokenizer(
        models.WordLevel(
            {"<unk>": 0, "<s>": 1, "</s>": 2, **{str(i): i for i in range(3, 64)}},
            unk_token="<unk>",
        )
    )
    tokenizer.save(str(directory / "tokenizer.json"))
    splits = {}
    for split, rows in (("train", 7), ("validation", 3)):
        ids = np.arange(rows * 8, dtype=np.uint16) % 61 + 3
        ids.astype("<u2").tofile(directory / f"{split}.bin")
        splits[split] = {"written_tokens": rows * 8, "sequences": rows, "bytes": rows * 16}
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "format": {"dtype": "<u2", "sequence_length": 8},
                "tokenizer": {
                    "file": "tokenizer.json",
                    "sha256": sha256_file(directory / "tokenizer.json"),
                },
                "splits": splits,
            }
        )
    )
    config = MiniLlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=8,
        bos_token_id=1,
        eos_token_id=2,
    )
    return directory, config


def test_accumulation_matches_full_uneven_batch(data):
    directory, config = data
    dataset = load_packed(directory, config)[0]["train"]
    context = ProcessContext(0, 1, torch.device("cpu"))
    recipe = TrainRecipe(total_tokens=56, global_batch_size=6, micro_batch_size=2, precision="fp32")
    torch.manual_seed(5)
    left = MiniLlamaForCausalLM(config)
    right = MiniLlamaForCausalLM(config)
    right.load_state_dict(left.state_dict())
    indices = np.array([1, 4, 6])
    loss_a, _ = train_update(
        left, make_optimizer(left, recipe, False), dataset, indices, recipe, context
    )
    full = replace(recipe, micro_batch_size=3)
    loss_b, _ = train_update(
        right, make_optimizer(right, full, False), dataset, indices, full, context
    )
    assert loss_a == pytest.approx(loss_b, abs=1e-6)
    for a, b in zip(left.parameters(), right.parameters()):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)


def test_resume_is_identical_and_rejects_data_changes(data, tmp_path):
    directory, config = data
    context = ProcessContext(0, 1, torch.device("cpu"))
    recipe = TrainRecipe(total_tokens=56, global_batch_size=4, micro_batch_size=2, precision="fp32")
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    run(config, recipe, directory, full, context)
    run(config, recipe, directory, resumed, context, stop_after_steps=1)
    run(config, recipe, directory, resumed, context, resume=resumed / "checkpoint.pt")
    expected = torch.load(full / "checkpoint.pt", weights_only=True)
    actual = torch.load(resumed / "checkpoint.pt", weights_only=True)
    for name in expected["model_state_dict"]:
        torch.testing.assert_close(
            actual["model_state_dict"][name], expected["model_state_dict"][name], rtol=0, atol=0
        )
    assert actual["progress"]["tokens_seen"] == 56
    assert actual["scheduler_state_dict"] == expected["scheduler_state_dict"]
    ids = np.fromfile(directory / "train.bin", dtype="<u2")
    ids[0] = 8
    ids.tofile(directory / "train.bin")
    with pytest.raises(ValueError, match="mismatch"):
        run(config, recipe, directory, resumed, context, resume=resumed / "checkpoint.pt")


def test_rejects_budget_larger_than_unique_data(data, tmp_path):
    directory, config = data
    with pytest.raises(ValueError, match="unique"):
        run(
            config,
            TrainRecipe(total_tokens=80, global_batch_size=4, micro_batch_size=2, precision="fp32"),
            directory,
            tmp_path / "bad",
            ProcessContext(0, 1, torch.device("cpu")),
        )
