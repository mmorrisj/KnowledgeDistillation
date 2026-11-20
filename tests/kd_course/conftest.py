import pytest

WORDS = [
    "<pad>",
    "<eos>",
    "<unk>",
    "the",
    "cat",
    "sat",
    "on",
    "mat",
    "a",
    "dog",
    "ran",
    "to",
    "floor",
]


@pytest.fixture(scope="session")
def tiny_lm():
    """A 2-layer GPT-2 with random weights and a 13-word vocabulary. No download needed."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {word: index for index, word in enumerate(WORDS)}
    raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )
    config = transformers.GPT2Config(
        vocab_size=len(WORDS),
        n_layer=2,
        n_head=2,
        n_embd=32,
        n_positions=64,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=0,
    )
    # Seeded, so every run starts from the same weights. Without this, tests that train the
    # model depended on the luck of the starting point and failed now and then.
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(config).eval()
    return model, tokenizer
