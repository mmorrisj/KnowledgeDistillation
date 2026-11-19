import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from kd_course.classifier import predict, train_classifier  # noqa: E402
from kd_course.data import LabelledSet  # noqa: E402

WORDS = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "card",
    "lost",
    "stolen",
    "rate",
    "exchange",
    "euro",
    "my",
    "the",
]


@pytest.fixture(scope="module")
def tiny_encoder_dir(tmp_path_factory):
    """A 2-layer BERT with random weights and a 12-word vocabulary, saved to disk."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing

    folder = tmp_path_factory.mktemp("tiny-bert")
    raw = Tokenizer(WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    raw.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    config = transformers.BertConfig(
        vocab_size=len(WORDS),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    torch.manual_seed(0)
    transformers.BertModel(config).save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return str(folder)


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


def test_training_learns_a_separable_toy_task(tiny_encoder_dir):
    trained = train_classifier(
        tiny_encoder_dir,
        TRAIN,
        TRAIN,
        num_labels=2,
        epochs=30,
        learning_rate=5e-3,
        batch_size=4,
        device="cpu",
        seed=0,
    )
    assert predict(trained, TRAIN.texts) == TRAIN.labels
    assert trained.history[-1]["train_loss"] < trained.history[0]["train_loss"]


def test_history_has_one_entry_per_epoch_with_validation_accuracy(tiny_encoder_dir):
    trained = train_classifier(
        tiny_encoder_dir, TRAIN, TRAIN, num_labels=2, epochs=3, device="cpu", seed=0
    )
    assert [h["epoch"] for h in trained.history] == [1, 2, 3]
    assert all(0 <= h["val_accuracy"] <= 1 for h in trained.history)


def test_same_seed_gives_the_same_predictions(tiny_encoder_dir):
    def run():
        trained = train_classifier(
            tiny_encoder_dir, TRAIN, TRAIN, num_labels=2, epochs=2, device="cpu", seed=7
        )
        return predict(trained, ["my card", "euro rate", "the the the"])

    assert run() == run()


def test_predict_handles_an_empty_list(tiny_encoder_dir):
    trained = train_classifier(
        tiny_encoder_dir, TRAIN, TRAIN, num_labels=2, epochs=1, device="cpu", seed=0
    )
    assert predict(trained, []) == []


def test_training_rejects_an_empty_training_set(tiny_encoder_dir):
    with pytest.raises(ValueError, match="empty"):
        train_classifier(tiny_encoder_dir, LabelledSet(), TRAIN, num_labels=2, device="cpu")


def test_training_rejects_a_label_outside_the_range(tiny_encoder_dir):
    bad = LabelledSet(texts=["my card"], labels=[5])
    with pytest.raises(ValueError, match="label"):
        train_classifier(tiny_encoder_dir, bad, TRAIN, num_labels=2, device="cpu")
