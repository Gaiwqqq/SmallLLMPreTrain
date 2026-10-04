"""用不同实现验证模型数学和 assistant mask，避免只测试自我一致。"""

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from dummym.checkpoint.huggingface import to_huggingface
from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM
from dummym.posttrain.sft import assert_assistant_mask
from dummym.tokenizer.train import CHAT_TEMPLATE


def test_reference_matches_transformers_logits_loss_and_gradients():
    torch.manual_seed(9)
    config = MiniLlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    reference = MiniLlamaForCausalLM(config)
    model = to_huggingface(reference)
    ids = torch.tensor([[1, 3, 7, 9, 2], [1, 4, 8, 10, 2]])
    labels = ids.clone()
    labels[0, 1] = -100
    expected, actual = reference(ids, labels=labels), model(ids, labels=labels)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=1e-5, atol=1e-6)
    expected.loss.backward()
    actual.loss.backward()
    for name, parameter in reference.named_parameters():
        hf_name = name if name.startswith("lm_head.") else "model." + name
        torch.testing.assert_close(
            model.get_parameter(hf_name).grad, parameter.grad, rtol=1e-4, atol=1e-6
        )


def test_chat_template_supervises_only_assistant():
    core = Tokenizer(
        models.WordLevel({"<unk>": 0, "Hello": 1, "Good": 2, "morning": 3}, unk_token="<unk>")
    )
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core, unk_token="<unk>", chat_template=CHAT_TEMPLATE
    )
    assert_assistant_mask(tokenizer)
