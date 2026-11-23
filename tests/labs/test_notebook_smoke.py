"""Run every lab notebook's own code cells on CPU, with tiny models and toy data swapped in.

This cannot check speed, memory or model quality: that needs the real models on a GPU.
It does catch wiring mistakes in the notebooks themselves: a wrong function name, a
missing argument, a variable used before it exists, a result one lab expects another
to have saved.

The labs run in order and share one results folder, because later labs read what
earlier ones saved.
"""

from __future__ import annotations

import copy
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
nbformat = pytest.importorskip("nbformat")
pytest.importorskip("peft")
pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

LABS = Path(__file__).resolve().parents[2] / "labs"
NOTEBOOKS = sorted(LABS.rglob("*.ipynb"))

LABELS = ["lost_card", "exchange_rate", "top_up"]
PHRASES = {
    0: ["my card lost", "the card stolen", "lost my card today", "card stolen help"],
    1: ["exchange rate euro", "the euro rate", "what rate abroad", "rate for euro"],
    2: ["top up failed", "my top up pending", "top up by card", "how to top up"],
}
WORDS = sorted(
    {word for phrases in PHRASES.values() for phrase in phrases for word in phrase.split()}
)


def toy_texts(n_per_class: int, tag: str) -> tuple[list[str], list[int]]:
    """Short banking-like messages. A number makes each one distinct."""
    texts, labels = [], []
    for label, phrases in PHRASES.items():
        for i in range(n_per_class):
            texts.append(f"{phrases[i % len(phrases)]} {tag} {i}")
            labels.append(label)
    return texts, labels


def word_tokenizer(
    special: dict[str, str], template: str | None = None, extra: list[str] | None = None
):
    """A tokenizer that knows the toy vocabulary and splits on spaces."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing

    vocab = list(dict.fromkeys(list(special.values()) + WORDS + LABELS + (extra or [])))
    raw = Tokenizer(
        WordLevel({word: i for i, word in enumerate(vocab)}, unk_token=special["unk_token"])
    )
    raw.pre_tokenizer = Whitespace()
    if template:
        ids = [(token, vocab.index(token)) for token in special.values() if token in template]
        raw.post_processor = TemplateProcessing(single=template, special_tokens=ids)
    return transformers.PreTrainedTokenizerFast(tokenizer_object=raw, **special), len(vocab)


@pytest.fixture(scope="session")
def tiny_lm():
    """A stand-in for the Qwen teacher and students: a 2-layer GPT-2 with a chat format."""
    tokenizer, size = word_tokenizer(
        {"pad_token": "<pad>", "eos_token": "<eos>", "unk_token": "<unk>"}
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    config = transformers.GPT2Config(
        vocab_size=size,
        n_layer=2,
        n_head=2,
        n_embd=32,
        n_positions=2048,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=0,
    )
    torch.manual_seed(0)
    return transformers.GPT2LMHeadModel(config).eval(), tokenizer


@pytest.fixture(scope="session")
def tiny_encoder_dir(tmp_path_factory):
    """A stand-in for ModernBERT and the embedding models: a 2-layer BERT saved to disk."""
    folder = tmp_path_factory.mktemp("smoke-encoder")
    special = {
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
    }
    tokenizer, size = word_tokenizer(special, template="[CLS] $A [SEP]")
    config = transformers.BertConfig(
        vocab_size=size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    transformers.BertModel(config).save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return str(folder)


@pytest.fixture(scope="session")
def tiny_vit_dir(tmp_path_factory):
    """A stand-in for the vision students."""
    folder = tmp_path_factory.mktemp("smoke-vit")
    config = transformers.ViTConfig(
        image_size=32,
        patch_size=8,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
    )
    torch.manual_seed(0)
    transformers.ViTModel(config).save_pretrained(folder)
    transformers.ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(folder)
    return str(folder)


@pytest.fixture(scope="session")
def tiny_clip():
    """A stand-in for the CLIP teacher."""
    special = {"pad_token": "<pad>", "eos_token": "<eos>", "unk_token": "<unk>"}
    tokenizer, size = word_tokenizer(special, template="$A <eos>", extra=["red", "blue", "thing"])
    text = transformers.CLIPTextConfig(
        vocab_size=size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
        eos_token_id=1,
        pad_token_id=0,
        bos_token_id=1,
    )
    vision = transformers.CLIPVisionConfig(
        image_size=32,
        patch_size=8,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
    )
    config = transformers.CLIPConfig(
        text_config=text.to_dict(), vision_config=vision.to_dict(), projection_dim=16
    )
    torch.manual_seed(0)
    processor = transformers.CLIPImageProcessor(
        size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}
    )
    return transformers.CLIPModel(config).eval(), processor, tokenizer


@pytest.fixture(scope="session")
def results_dir(tmp_path_factory):
    """One results folder for the whole run, so later labs find what earlier ones saved."""
    return tmp_path_factory.mktemp("smoke-results")


def colour_image(label: int, seed: int) -> Image.Image:
    generator = torch.Generator().manual_seed(seed)
    pixels = torch.randint(0, 60, (32, 32, 3), generator=generator, dtype=torch.uint8)
    pixels[..., 0 if label == 0 else 2] += 180
    return Image.fromarray(pixels.numpy(), mode="RGB")


@pytest.fixture
def swapped(monkeypatch, tiny_lm, tiny_encoder_dir, tiny_vit_dir, tiny_clip, results_dir):
    """Replace everything that needs a GPU, a download or a network with a small local stand-in."""
    import httpx

    import kd_course.classifier as classifier
    import kd_course.data as data
    import kd_course.distill as distill
    import kd_course.embeddings as embeddings
    import kd_course.env as env
    import kd_course.generative as generative
    import kd_course.lora as lora
    import kd_course.models as models
    import kd_course.titles as titles
    import kd_course.vision as vision

    monkeypatch.setenv("KD_COURSE_RESULTS_DIR", str(results_dir))
    monkeypatch.delenv("KD_COURSE_DATABASE_URL", raising=False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    # The machine check: pretend this is a ready A10G.
    ready = env.EnvReport(
        python="3", torch="2", gpu=env.GpuInfo("NVIDIA A10G", 22.5, True), free_disk_gb=100.0
    )
    monkeypatch.setattr(env, "check_environment", lambda *args, **kwargs: ready)

    # Text data: three types, small splits.
    train_texts, train_labels = toy_texts(20, "train")
    test_texts, test_labels = toy_texts(12, "test")
    bundle = data.DatasetBundle(
        train=data.LabelledSet(train_texts, train_labels),
        test=data.LabelledSet(test_texts, test_labels),
        label_names=LABELS,
    )
    real_splits = data.make_splits
    monkeypatch.setattr(data, "load_banking77", lambda *args, **kwargs: copy.deepcopy(bundle))
    monkeypatch.setattr(
        data,
        "make_splits",
        lambda train, test, **kwargs: real_splits(
            train, test, labels_per_class=4, val_per_class=2, test_size=30
        ),
    )

    # Language models: the tiny one, a fresh copy per load so labs can train and delete freely.
    base_lm, lm_tokenizer = tiny_lm

    def load_lm(*args, **kwargs):
        return copy.deepcopy(base_lm), lm_tokenizer

    monkeypatch.setattr(models, "load_causal_lm", load_lm)
    monkeypatch.setattr(models, "load_causal_lm_4bit", load_lm)

    real_add_lora = lora.add_lora
    monkeypatch.setattr(
        lora,
        "add_lora",
        lambda model, rank=16, alpha=32.0, **kwargs: real_add_lora(
            model, rank=rank, alpha=alpha, target_modules=["c_attn"]
        ),
    )

    class MeasuredOnCpu(generative.measure_training):
        """Reports a made-up peak memory figure, since a CPU has none to report."""

        def __exit__(self, *exc):
            super().__exit__(*exc)
            self.cost = generative.TrainingCost(seconds=self.cost.seconds, peak_memory_gb=1.0)

    monkeypatch.setattr(generative, "measure_training", MeasuredOnCpu)

    # Encoder students: the tiny one, on CPU, one epoch.
    def on_cpu(function, model_dir):
        def wrapper(model_id, *args, **kwargs):
            kwargs.update(device="cpu", epochs=1)
            return function(model_dir, *args, **kwargs)

        return wrapper

    monkeypatch.setattr(
        classifier, "train_classifier", on_cpu(classifier.train_classifier, tiny_encoder_dir)
    )
    monkeypatch.setattr(
        distill,
        "train_distilled_classifier",
        on_cpu(distill.train_distilled_classifier, tiny_encoder_dir),
    )

    # Titles: a random tiny model cannot write a usable title, so the teacher's titles are scripted.
    monkeypatch.setattr(
        titles,
        "generate_titles",
        lambda teacher, messages, **kwargs: [
            " ".join(message.split()[:3]) + " request" for message in messages
        ],
    )

    # Images: two colours standing in for 37 breeds.
    def toy_images(n_per_class, offset):
        labels = [label for label in (0, 1) for _ in range(n_per_class)]
        return vision.ImageSet(
            images=[colour_image(label, offset + i) for i, label in enumerate(labels)],
            labels=labels,
            ids=[f"img{offset + i}" for i in range(len(labels))],
        )

    pets = vision.ImageBundle(
        train=toy_images(14, 0), test=toy_images(8, 500), label_names=["red_thing", "blue_thing"]
    )
    real_image_splits = vision.make_image_splits
    monkeypatch.setattr(vision, "load_pets", lambda *args, **kwargs: pets)
    monkeypatch.setattr(
        vision,
        "make_image_splits",
        lambda train, test, **kwargs: real_image_splits(
            train, test, labels_per_class=3, val_per_class=2, test_size=12
        ),
    )
    monkeypatch.setattr(vision, "load_zero_shot_teacher", lambda *args, **kwargs: tiny_clip)
    monkeypatch.setattr(
        vision, "train_image_classifier", on_cpu(vision.train_image_classifier, tiny_vit_dir)
    )
    monkeypatch.setattr(models, "resolve_model_path", lambda model_id, model_dir=None: tiny_vit_dir)

    # Embedding models: the tiny encoder.
    def load_embedder(*args, **kwargs):
        return (
            transformers.AutoModel.from_pretrained(tiny_encoder_dir).eval(),
            transformers.AutoTokenizer.from_pretrained(tiny_encoder_dir),
        )

    real_embedding_training = embeddings.train_embedding_student
    monkeypatch.setattr(embeddings, "load_embedder", load_embedder)
    monkeypatch.setattr(
        embeddings,
        "train_embedding_student",
        lambda *args, **kwargs: real_embedding_training(
            *args, **{**kwargs, "device": "cpu", "epochs": 1, "batch_size": 16}
        ),
    )

    # The student service: answer through the real web app, in-process, with no port.
    def service_client(*args, **kwargs):
        from fastapi.testclient import TestClient

        from kd_course.results import default_results_dir
        from kd_course.serving.app import create_student_app
        from kd_course.serving.export import load_classifier

        predictor, labels = load_classifier(
            default_results_dir() / "students" / "classifier", device="cpu"
        )
        return TestClient(create_student_app(predictor, labels))

    monkeypatch.setattr(httpx, "Client", service_client)

    # Notebooks show images with IPython, which the test environment may not have.
    if "IPython" not in sys.modules:
        display = types.ModuleType("IPython.display")
        display.display = lambda *args, **kwargs: None
        package = types.ModuleType("IPython")
        package.display = display
        monkeypatch.setitem(sys.modules, "IPython", package)
        monkeypatch.setitem(sys.modules, "IPython.display", display)


def test_every_lab_has_a_smoke_test():
    assert len(NOTEBOOKS) >= 9


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.parent.name)
def test_notebook_cells_run(path, swapped):
    notebook = nbformat.read(path, as_version=4)
    namespace: dict = {"__name__": "__main__"}
    cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    for number, source in enumerate(cells, start=1):
        try:
            exec(compile(source, f"{path.name}, code cell {number}", "exec"), namespace)  # noqa: S102
        except BaseException as error:
            first_lines = "\n".join(source.splitlines()[:6])
            pytest.fail(
                f"{path.relative_to(LABS)}: code cell {number} of {len(cells)} failed with "
                f"{type(error).__name__}: {error}\n\n{first_lines}\n...",
                pytrace=False,
            )
