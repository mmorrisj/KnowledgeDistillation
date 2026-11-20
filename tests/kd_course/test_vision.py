import math
from collections import Counter

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
PIL = pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from kd_course.vision import (  # noqa: E402
    ImageSet,
    make_image_splits,
    predict_images,
    prompt_for,
    to_pixels,
    train_image_classifier,
    zero_shot_scores,
)


def colour_image(label: int, seed: int) -> Image.Image:
    """A 32x32 image that is mostly red for class 0 and mostly blue for class 1."""
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randint(0, 60, (32, 32, 3), generator=generator, dtype=torch.uint8)
    noise[..., 0 if label == 0 else 2] += 180
    return Image.fromarray(noise.numpy(), mode="RGB")


def toy_set(n_per_class: int, offset: int = 0) -> ImageSet:
    labels = [label for label in (0, 1) for _ in range(n_per_class)]
    images = [colour_image(label, offset + i) for i, label in enumerate(labels)]
    return ImageSet(
        images=images, labels=labels, ids=[f"img{offset + i}" for i in range(len(labels))]
    )


@pytest.fixture(scope="module")
def tiny_vit_dir(tmp_path_factory):
    folder = tmp_path_factory.mktemp("tiny-vit")
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


@pytest.fixture(scope="module")
def tiny_clip():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing

    words = [
        "<pad>",
        "<eos>",
        "<unk>",
        "a",
        "photo",
        "of",
        "red",
        "blue",
        "thing",
        ",",
        "type",
        "pet",
    ]
    raw = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    raw.post_processor = TemplateProcessing(single="$A <eos>", special_tokens=[("<eos>", 1)])
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )
    text = transformers.CLIPTextConfig(
        vocab_size=len(words),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=16,
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
    model = transformers.CLIPModel(config).eval()
    processor = transformers.CLIPImageProcessor(
        size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}
    )
    return model, processor, tokenizer


def test_prompt_uses_spaces_in_place_of_underscores():
    assert prompt_for("american_bulldog", "a photo of a {}") == "a photo of a american bulldog"


def test_splits_have_the_requested_sizes_and_do_not_overlap():
    splits = make_image_splits(
        toy_set(20), toy_set(10, offset=100), labels_per_class=4, val_per_class=2, test_size=10
    )
    assert Counter(splits.gold_train.labels) == {0: 4, 1: 4}
    assert Counter(splits.gold_val.labels) == {0: 2, 1: 2}
    assert len(splits.pool.labels) == 40 - 8 - 4
    assert len(splits.test.labels) == 10
    groups = [set(s.ids) for s in (splits.gold_train, splits.gold_val, splits.pool, splits.test)]
    assert sum(len(g) for g in groups) == len(set().union(*groups))


def test_splits_keep_images_labels_and_ids_lined_up():
    data = toy_set(20)
    splits = make_image_splits(data, toy_set(10, offset=100), test_size=10)
    by_id = dict(zip(data.ids, data.labels, strict=True))
    assert all(
        by_id[i] == label for i, label in zip(splits.pool.ids, splits.pool.labels, strict=True)
    )
    assert len(splits.pool.images) == len(splits.pool.labels)
    assert isinstance(splits.pool.images[0], Image.Image)


def test_splits_are_repeatable_and_change_with_the_seed():
    a = make_image_splits(toy_set(20), toy_set(10, offset=100), test_size=10, seed=13)
    b = make_image_splits(toy_set(20), toy_set(10, offset=100), test_size=10, seed=13)
    c = make_image_splits(toy_set(20), toy_set(10, offset=100), test_size=10, seed=14)
    assert a.gold_train.ids == b.gold_train.ids
    assert a.gold_train.ids != c.gold_train.ids


def test_a_class_with_too_few_images_is_an_error():
    with pytest.raises(ValueError, match="class 0"):
        make_image_splits(
            toy_set(3), toy_set(10, offset=100), labels_per_class=4, val_per_class=2, test_size=10
        )


def test_to_pixels_gives_one_tensor_row_per_image(tiny_clip):
    _, processor, _ = tiny_clip
    pixels = to_pixels(processor, toy_set(3).images, batch_size=4)
    assert pixels.shape == (6, 3, 32, 32)
    assert to_pixels(processor, []).shape[0] == 0


def test_zero_shot_scores_match_the_model_and_add_up_to_one(tiny_clip):
    model, processor, tokenizer = tiny_clip
    images = toy_set(3).images
    labels = ["red_thing", "blue_thing"]
    scores = zero_shot_scores(
        model, processor, tokenizer, images, labels, template="a photo of a {}", batch_size=4
    )
    assert len(scores) == 6
    for row in scores:
        assert sum(math.exp(v) for v in row) == pytest.approx(1.0, abs=1e-5)

    text = tokenizer(
        ["a photo of a red thing", "a photo of a blue thing"], padding=True, return_tensors="pt"
    )
    pixels = processor(images=images, return_tensors="pt")["pixel_values"]
    with torch.inference_mode():
        expected = torch.log_softmax(model(pixel_values=pixels, **text).logits_per_image, dim=-1)
    assert torch.tensor(scores) == pytest.approx(expected, abs=1e-4)


def test_zero_shot_scores_of_no_images_is_empty(tiny_clip):
    model, processor, tokenizer = tiny_clip
    assert zero_shot_scores(model, processor, tokenizer, [], ["red_thing"]) == []


def pixels_for(folder, image_set):
    processor = transformers.AutoImageProcessor.from_pretrained(folder)
    return to_pixels(processor, image_set.images)


def test_a_student_learns_from_human_labels(tiny_vit_dir):
    data = toy_set(12)
    pixels = pixels_for(tiny_vit_dir, data)
    trained = train_image_classifier(
        tiny_vit_dir,
        pixels,
        data.labels,
        pixels,
        data.labels,
        num_labels=2,
        epochs=15,
        learning_rate=3e-3,
        batch_size=8,
        device="cpu",
        seed=0,
    )
    assert predict_images(trained, pixels) == data.labels
    assert [row["epoch"] for row in trained.history] == list(range(1, 16))


def confident(label: int) -> list[float]:
    return [math.log(0.9), math.log(0.1)] if label == 0 else [math.log(0.1), math.log(0.9)]


def test_a_student_learns_from_teacher_probabilities_alone(tiny_vit_dir):
    data = toy_set(12)
    pixels = pixels_for(tiny_vit_dir, data)
    trained = train_image_classifier(
        tiny_vit_dir,
        pixels,
        [None] * len(data.labels),
        pixels,
        data.labels,
        num_labels=2,
        teacher_log_probs=[confident(label) for label in data.labels],
        temperature=1.0,
        alpha=1.0,
        epochs=15,
        learning_rate=3e-3,
        batch_size=8,
        device="cpu",
        seed=0,
    )
    assert predict_images(trained, pixels) == data.labels


def test_lora_trains_only_adapters_and_the_new_output_layer(tiny_vit_dir):
    pytest.importorskip("peft")
    data = toy_set(8)
    pixels = pixels_for(tiny_vit_dir, data)
    trained = train_image_classifier(
        tiny_vit_dir,
        pixels,
        data.labels,
        pixels,
        data.labels,
        num_labels=2,
        epochs=2,
        learning_rate=3e-3,
        batch_size=8,
        device="cpu",
        seed=0,
        lora_rank=2,
    )
    trainable = [name for name, p in trained.model.named_parameters() if p.requires_grad]
    assert trainable
    assert all("lora" in name or "classifier" in name for name in trainable)
    assert len(predict_images(trained, pixels)) == len(data.labels)


def test_missing_labels_without_a_teacher_is_an_error(tiny_vit_dir):
    data = toy_set(4)
    pixels = pixels_for(tiny_vit_dir, data)
    with pytest.raises(ValueError, match="teacher"):
        train_image_classifier(
            tiny_vit_dir, pixels, [None] * 8, pixels, data.labels, num_labels=2, device="cpu"
        )


def test_misaligned_inputs_are_an_error(tiny_vit_dir):
    data = toy_set(4)
    pixels = pixels_for(tiny_vit_dir, data)
    with pytest.raises(ValueError, match="length"):
        train_image_classifier(
            tiny_vit_dir, pixels, [0, 1], pixels, data.labels, num_labels=2, device="cpu"
        )


def test_predict_images_of_nothing_is_empty(tiny_vit_dir):
    data = toy_set(4)
    pixels = pixels_for(tiny_vit_dir, data)
    trained = train_image_classifier(
        tiny_vit_dir, pixels, data.labels, pixels, data.labels, num_labels=2, epochs=1, device="cpu"
    )
    assert predict_images(trained, pixels[:0]) == []


def test_load_pets_from_a_local_folder(tmp_path):
    datasets = pytest.importorskip("datasets")
    from kd_course.vision import load_pets

    features = datasets.Features(
        {
            "image": datasets.Image(),
            "label": datasets.ClassLabel(names=["abyssinian", "beagle"]),
            "image_id": datasets.Value("string"),
        }
    )
    folder = tmp_path / "oxford-iiit-pet"
    folder.mkdir()
    for split, n in {"train": 3, "test": 2}.items():
        rows = {
            "image": [colour_image(i % 2, i) for i in range(n)],
            "label": [i % 2 for i in range(n)],
            "image_id": [f"{split}_{i}" for i in range(n)],
        }
        datasets.Dataset.from_dict(rows, features=features).to_parquet(
            str(folder / f"{split}.parquet")
        )
    bundle = load_pets(data_dir=str(tmp_path))
    assert bundle.label_names == ["abyssinian", "beagle"]
    assert bundle.train.labels == [0, 1, 0]
    assert bundle.test.ids == ["test_0", "test_1"]
    assert bundle.train.images[0].mode == "RGB"
    assert len(bundle.train.images) == 3


def test_attention_targets_are_found_for_the_installed_library(tiny_vit_dir):
    from kd_course.vision import attention_targets

    model = transformers.AutoModelForImageClassification.from_pretrained(tiny_vit_dir, num_labels=2)
    targets = attention_targets(model)
    assert targets in (["q_proj", "v_proj"], ["query", "value"])


def test_attention_targets_error_names_the_way_out():
    from kd_course.vision import attention_targets

    with pytest.raises(ValueError, match="lora_targets"):
        attention_targets(torch.nn.Linear(2, 2))


def test_zero_shot_or_load_scores_once_then_reads_the_file(tiny_clip, tmp_path):
    from kd_course.vision import zero_shot_or_load

    model, processor, tokenizer = tiny_clip
    data = toy_set(2)
    path = tmp_path / "teacher_scores" / "pool.json"
    first = zero_shot_or_load(path, data, model, processor, tokenizer, ["red_thing", "blue_thing"])
    assert path.is_file()
    # With no model, the second call can only succeed by reading the saved file.
    assert zero_shot_or_load(path, data, None, None, None, ["red_thing", "blue_thing"]) == first


def test_zero_shot_or_load_refuses_scores_saved_for_other_images(tiny_clip, tmp_path):
    from kd_course.vision import zero_shot_or_load

    model, processor, tokenizer = tiny_clip
    path = tmp_path / "pool.json"
    zero_shot_or_load(path, toy_set(2), model, processor, tokenizer, ["red_thing", "blue_thing"])
    with pytest.raises(ValueError, match="different"):
        zero_shot_or_load(
            path, toy_set(2, offset=50), None, None, None, ["red_thing", "blue_thing"]
        )
