import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from kd_course.lora import LoRALinear, add_lora, count_parameters_by_kind, merge_lora  # noqa: E402


def test_scratch_lora_starts_as_an_exact_copy_of_the_original_layer():
    torch.manual_seed(0)
    base = torch.nn.Linear(16, 8)
    layer = LoRALinear(base, rank=4, alpha=8)
    x = torch.randn(3, 16)
    assert torch.allclose(layer(x), base(x))


def test_scratch_lora_trains_only_the_two_small_matrices():
    layer = LoRALinear(torch.nn.Linear(16, 8), rank=4, alpha=8)
    trainable = {name for name, p in layer.named_parameters() if p.requires_grad}
    assert trainable == {"lora_a", "lora_b"}
    assert sum(p.numel() for p in layer.parameters() if p.requires_grad) == 4 * (16 + 8)


def test_scratch_lora_changes_the_output_once_trained():
    torch.manual_seed(0)
    base = torch.nn.Linear(16, 8)
    layer = LoRALinear(base, rank=4, alpha=8)
    x, target = torch.randn(32, 16), torch.randn(32, 8)
    optimizer = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=0.1)
    before = torch.nn.functional.mse_loss(layer(x), target).item()
    for _ in range(50):
        loss = torch.nn.functional.mse_loss(layer(x), target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    assert torch.nn.functional.mse_loss(layer(x), target).item() < before
    assert torch.equal(base.weight, layer.base.weight)  # the original weights never moved


def test_scratch_lora_merged_layer_gives_the_same_output():
    torch.manual_seed(0)
    layer = LoRALinear(torch.nn.Linear(16, 8), rank=4, alpha=8)
    with torch.no_grad():
        layer.lora_b.normal_()
    x = torch.randn(5, 16)
    merged = layer.merged()
    assert isinstance(merged, torch.nn.Linear)
    assert torch.allclose(merged(x), layer(x), atol=1e-5)


def test_add_lora_leaves_the_base_frozen_and_adds_a_small_trainable_part(tiny_lm):
    import copy

    model = copy.deepcopy(tiny_lm[0])
    total_before = sum(p.numel() for p in model.parameters())
    lora_model = add_lora(model, rank=4, alpha=8, target_modules=["c_attn"])
    counts = count_parameters_by_kind(lora_model)
    assert counts["total"] == total_before + counts["trainable"]
    assert 0 < counts["trainable"] < total_before * 0.2
    assert counts["trainable_share"] == pytest.approx(counts["trainable"] / counts["total"])


def test_count_parameters_by_kind_on_a_plain_model_is_all_trainable(tiny_lm):
    counts = count_parameters_by_kind(tiny_lm[0])
    assert counts["trainable"] == counts["total"]


def test_merge_lora_returns_a_plain_model_with_the_same_outputs(tiny_lm):
    import copy

    model = copy.deepcopy(tiny_lm[0])
    lora_model = add_lora(model, rank=4, alpha=8, target_modules=["c_attn"])
    with torch.no_grad():
        for name, parameter in lora_model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.05)
    lora_model.eval()  # switch dropout off before comparing outputs
    ids = torch.tensor([[3, 4, 5, 6]])
    with torch.inference_mode():
        expected = lora_model(ids).logits
    merged = merge_lora(lora_model)
    assert not any("lora" in name for name, _ in merged.named_parameters())
    with torch.inference_mode():
        assert torch.allclose(merged(ids).logits, expected, atol=1e-4)


def test_add_lora_rejects_a_rank_below_one(tiny_lm):
    with pytest.raises(ValueError, match="rank"):
        add_lora(tiny_lm[0], rank=0)


def test_folder_size_adds_up_every_file(tmp_path):
    from kd_course.lora import folder_size_mb

    (tmp_path / "a.bin").write_bytes(b"x" * 1_048_576)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"x" * 524_288)
    assert folder_size_mb(tmp_path) == pytest.approx(1.5)


def test_folder_size_of_a_missing_folder_is_zero(tmp_path):
    from kd_course.lora import folder_size_mb

    assert folder_size_mb(tmp_path / "none") == 0.0
