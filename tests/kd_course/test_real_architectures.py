"""The lab code against tiny, randomly initialised copies of the real model designs.

The other tests use GPT-2 and BERT stand-ins. The labs use Qwen2 and ModernBERT, which
differ in ways that matter: layer names, how attention handles saved state, and what a
tokenizer returns. These tests build each design at toy size and run the lab code on it.
They check that the code works with the design, not that the real models perform well.
"""

from __future__ import annotations

import copy
import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from kd_course.data import LabelledSet  # noqa: E402

WORDS = [
    "<pad>",
    "<|im_end|>",
    "<unk>",
    "<|im_start|>",
    "user",
    "assistant",
    "my",
    "card",
    "lost",
    "stolen",
    "the",
    "euro",
    "rate",
    "exchange",
    "lost_card",
    "exchange_rate",
    "Message",
    "Type",
    ":",
    "types",
]
CHATML = (
    "{% for message in messages %}"
    "<|im_start|> {{ message['role'] }} {{ message['content'] }} <|im_end|> "
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|> assistant {% endif %}"
)
LABELS = ["lost_card", "exchange_rate"]
TRAIN = LabelledSet(
    texts=[
        "my card lost",
        "the card stolen",
        "lost my card",
        "stolen card",
        "exchange rate euro",
        "the euro rate",
        "rate exchange",
        "euro exchange rate",
    ],
    labels=[0, 0, 0, 0, 1, 1, 1, 1],
)


def word_tokenizer(**special):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit

    raw = Tokenizer(WordLevel({word: i for i, word in enumerate(WORDS)}, unk_token="<unk>"))
    raw.pre_tokenizer = WhitespaceSplit()
    return transformers.PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>", **special)


@pytest.fixture(scope="module")
def tiny_qwen():
    """Qwen2 at toy size, with a chat format shaped like the real one."""
    tokenizer = word_tokenizer(pad_token="<pad>", eos_token="<|im_end|>")
    tokenizer.chat_template = CHATML
    config = transformers.Qwen2Config(
        vocab_size=len(WORDS),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=0,
    )
    torch.manual_seed(0)
    return transformers.Qwen2ForCausalLM(config).eval(), tokenizer


def test_default_lora_targets_exist_in_qwen2(tiny_qwen):
    from kd_course.lora import DEFAULT_TARGETS, add_lora, count_parameters_by_kind

    model = copy.deepcopy(tiny_qwen[0])
    names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    assert set(DEFAULT_TARGETS) <= names
    counts = count_parameters_by_kind(add_lora(model, rank=4, alpha=8))
    assert 0 < counts["trainable"] < counts["total"]


def test_cached_label_scoring_matches_one_pass_per_label_on_qwen2(tiny_qwen):
    from kd_course.label_scoring import score_labels
    from kd_course.prompted import build_classification_prompt

    model, tokenizer = tiny_qwen
    texts = ["my card lost", "the euro rate", "stolen card exchange"]
    fast = score_labels(model, tokenizer, texts, LABELS)
    for text, row in zip(texts, fast, strict=True):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": build_classification_prompt(text, LABELS)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(prompt)["input_ids"]
        scores = []
        for label in LABELS:
            label_ids = tokenizer(label, add_special_tokens=False)["input_ids"] + [
                tokenizer.eos_token_id
            ]
            with torch.inference_mode():
                logits = model(torch.tensor([prompt_ids + label_ids])).logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)
            start = len(prompt_ids)
            scores.append(sum(log_probs[start + j - 1, t].item() for j, t in enumerate(label_ids)))
        total = math.log(sum(math.exp(s) for s in scores))
        assert row == pytest.approx([s - total for s in scores], abs=1e-4)


def test_batched_generation_matches_one_at_a_time_on_qwen2(tiny_qwen):
    """Left padding and the attention mask must not change what a model writes."""
    from kd_course.teachers import LocalTeacher

    model, tokenizer = tiny_qwen
    prompts = ["my card", "the euro rate exchange my card lost", "stolen"]
    together = LocalTeacher(model, tokenizer, batch_size=3).complete(prompts, max_new_tokens=6)
    alone = [
        LocalTeacher(model, tokenizer, batch_size=1).complete([p], max_new_tokens=6)[0]
        for p in prompts
    ]
    assert together == alone


def test_lora_training_runs_on_qwen2_and_merging_keeps_its_answers(tiny_qwen):
    """Adapters on a frozen random model cannot learn much, so this checks the mechanics:
    the loss falls, only adapter weights move, and merging does not change the answers."""
    from kd_course.generative import predict_generative, train_generative_classifier
    from kd_course.lora import add_lora, merge_lora

    model = add_lora(copy.deepcopy(tiny_qwen[0]), rank=8, alpha=16)
    before_training = {name: p.detach().clone() for name, p in model.named_parameters()}
    history = train_generative_classifier(
        model, tiny_qwen[1], TRAIN, LABELS, epochs=20, learning_rate=1e-2, batch_size=4, seed=0
    )
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    moved = {
        name for name, p in model.named_parameters() if not torch.equal(p, before_training[name])
    }
    assert moved and all("lora" in name for name in moved)

    model.eval()
    answers = predict_generative(model, tiny_qwen[1], TRAIN.texts, LABELS)
    assert predict_generative(merge_lora(model), tiny_qwen[1], TRAIN.texts, LABELS) == answers


def test_a_fully_fine_tuned_qwen2_student_learns_the_toy_task(tiny_qwen):
    from kd_course.generative import predict_generative, train_generative_classifier

    model = copy.deepcopy(tiny_qwen[0])
    train_generative_classifier(
        model, tiny_qwen[1], TRAIN, LABELS, epochs=60, learning_rate=5e-3, batch_size=4, seed=0
    )
    assert predict_generative(model, tiny_qwen[1], TRAIN.texts, LABELS) == TRAIN.labels


def test_token_level_records_line_up_on_qwen2(tiny_qwen):
    from kd_course.token_kd import topk_targets, train_token_kd

    teacher, tokenizer = tiny_qwen
    prompts, answers = ["Message : my card lost Type :"], ["lost_card"]
    records = topk_targets(teacher, tokenizer, prompts, answers, k=5, answer_prefix="")
    assert records[0].target_ids[-1] == tokenizer.eos_token_id
    student = copy.deepcopy(teacher)
    history = train_token_kd(student, tokenizer, prompts * 4, records * 4, epochs=2, batch_size=2)
    assert all(math.isfinite(row["train_loss"]) for row in history)


@pytest.fixture(scope="module")
def tiny_modernbert_dir(tmp_path_factory):
    """ModernBERT at toy size, saved to disk."""
    folder = tmp_path_factory.mktemp("tiny-modernbert")
    tokenizer = word_tokenizer(pad_token="<pad>", cls_token="<|im_start|>", sep_token="<|im_end|>")
    tokenizer.model_input_names = ["input_ids", "attention_mask"]  # as the real tokenizer does
    config = transformers.ModernBertConfig(
        vocab_size=len(WORDS),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=3,
        cls_token_id=3,
        sep_token_id=1,
        global_attn_every_n_layers=1,
        local_attention=16,
        reference_compile=False,
    )
    torch.manual_seed(0)
    transformers.ModernBertModel(config).save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return str(folder)


def test_modernbert_trains_as_a_classifier(tiny_modernbert_dir):
    from kd_course.classifier import predict, train_classifier

    trained = train_classifier(
        tiny_modernbert_dir,
        TRAIN,
        TRAIN,
        num_labels=2,
        epochs=30,
        learning_rate=5e-3,
        batch_size=4,
        device="cpu",
        seed=0,
    )
    assert trained.history[-1]["train_loss"] < trained.history[0]["train_loss"]
    assert len(predict(trained, TRAIN.texts)) == len(TRAIN.texts)


def test_modernbert_trains_with_the_distillation_loss(tiny_modernbert_dir):
    from kd_course.distill import DistillSet, train_distilled_classifier

    confident = {0: [math.log(0.9), math.log(0.1)], 1: [math.log(0.1), math.log(0.9)]}
    data = DistillSet(
        texts=TRAIN.texts,
        teacher_log_probs=[confident[label] for label in TRAIN.labels],
        labels=[label if i % 2 else None for i, label in enumerate(TRAIN.labels)],
    )
    trained = train_distilled_classifier(
        tiny_modernbert_dir, data, TRAIN, num_labels=2, epochs=3, batch_size=4, device="cpu"
    )
    assert all(math.isfinite(row["train_loss"]) for row in trained.history)


def test_modernbert_can_be_saved_served_and_stored_in_8_bits(tiny_modernbert_dir, tmp_path):
    from kd_course.classifier import train_classifier
    from kd_course.serving.export import (
        load_classifier,
        model_size_mb,
        quantize_int8,
        save_classifier,
    )

    trained = train_classifier(
        tiny_modernbert_dir, TRAIN, TRAIN, num_labels=2, epochs=2, device="cpu"
    )
    folder = save_classifier(trained, LABELS, tmp_path / "student")
    full, labels = load_classifier(folder, device="cpu")
    assert labels == LABELS
    expected = [label for label, _ in full.predict(TRAIN.texts)]

    small = quantize_int8(trained.model)
    assert model_size_mb(small) < model_size_mb(trained.model)
    eight_bit, _ = load_classifier(folder, device="cpu", quantize=True)
    answers = eight_bit.predict(TRAIN.texts)
    assert len(answers) == len(TRAIN.texts)
    assert all(0 <= confidence <= 1 for _, confidence in answers)
    # Rounding can flip a borderline answer on a toy model; most must agree.
    agree = sum(a == b for (a, _), b in zip(answers, expected, strict=True))
    assert agree >= len(expected) - 2
