"""The Module 3 lab flow, end to end, on tiny CPU models. Checks the pieces fit together."""

import pytest

pytest.importorskip("torch")

from kd_course.classifier import predict, train_classifier  # noqa: E402
from kd_course.data import LabelledSet  # noqa: E402
from kd_course.distill import DistillSet, train_distilled_classifier  # noqa: E402
from kd_course.label_scoring import score_or_load, top_labels  # noqa: E402
from kd_course.metrics import accuracy, gap_closed, paired_bootstrap, summarize  # noqa: E402
from kd_course.results import format_results, load_results, require_result, save_result  # noqa: E402

from .test_classifier import TRAIN, tiny_encoder_dir  # noqa: E402, F401
from .test_vision import tiny_clip, tiny_vit_dir  # noqa: E402, F401


def test_white_box_lab_flow(tiny_lm, tiny_encoder_dir, tmp_path):  # noqa: F811
    teacher, teacher_tokenizer = tiny_lm
    labels = ["cat", "dog"]
    gold = LabelledSet(texts=TRAIN.texts[:4], labels=TRAIN.labels[:4])
    pool = LabelledSet(texts=TRAIN.texts[4:], labels=TRAIN.labels[4:])
    test = TRAIN

    args = {"model": teacher, "tokenizer": teacher_tokenizer, "label_names": labels}
    test_scores = score_or_load(tmp_path / "teacher_scores_test.json", texts=test.texts, **args)
    gold_scores = score_or_load(tmp_path / "teacher_scores_gold.json", texts=gold.texts, **args)
    pool_scores = score_or_load(tmp_path / "teacher_scores_pool.json", texts=pool.texts, **args)

    teacher_pred = top_labels(test_scores)
    save_result(
        "teacher_scored",
        summarize(test.labels, teacher_pred, 2, n_resamples=50),
        teacher_pred,
        tmp_path,
    )

    baseline = train_classifier(tiny_encoder_dir, gold, gold, num_labels=2, epochs=1, device="cpu")
    baseline_pred = predict(baseline, test.texts)
    save_result(
        "baseline_gold_616",
        summarize(test.labels, baseline_pred, 2, n_resamples=50),
        baseline_pred,
        tmp_path,
    )

    hard = LabelledSet(texts=gold.texts + pool.texts, labels=gold.labels + top_labels(pool_scores))
    hard_pred = predict(
        train_classifier(tiny_encoder_dir, hard, gold, num_labels=2, epochs=1, device="cpu"),
        test.texts,
    )

    soft = DistillSet(
        texts=gold.texts + pool.texts,
        teacher_log_probs=gold_scores + pool_scores,
        labels=gold.labels + [None] * len(pool),
    )
    student = train_distilled_classifier(
        tiny_encoder_dir, soft, gold, num_labels=2, epochs=1, device="cpu"
    )
    soft_pred = predict(student, test.texts)
    soft_result = summarize(test.labels, soft_pred, 2, n_resamples=50)
    save_result("distilled_probabilities_T2", soft_result, soft_pred, tmp_path)

    saved_baseline = require_result("baseline_gold_616", tmp_path)
    table = format_results(load_results(tmp_path))
    assert "teacher_scored" in table and "distilled_probabilities_T2" in table
    gap = gap_closed(
        saved_baseline["metrics"]["accuracy"],
        soft_result["accuracy"],
        accuracy(test.labels, teacher_pred),
    )
    assert gap is None or isinstance(gap, float)
    comparison = paired_bootstrap(
        accuracy, test.labels, saved_baseline["predictions"], soft_pred, n_resamples=50
    )
    assert comparison.low <= comparison.difference <= comparison.high
    assert len(hard_pred) == len(soft_pred) == len(test)


def test_black_box_lab_flow(tiny_encoder_dir, tmp_path):  # noqa: F811
    from kd_course.metrics import teacher_verdict
    from kd_course.synthetic import (
        dedupe,
        generate_examples,
        labelled_from_predictions,
        pick_examples,
        round_trip_filter,
    )
    from kd_course.teachers import classify_with_teacher

    labels = ["lost_card", "exchange_rate"]

    class KeywordTeacher:
        """A stand-in black-box teacher: answers by keyword, and can write new messages."""

        def complete(self, prompts, max_new_tokens=16):
            replies = []
            for prompt in prompts:
                if prompt.startswith("Write"):
                    kind = "card lost" if "lost_card" in prompt else "euro rate"
                    replies.append(f"1. my {kind}\n2. the {kind}\n3. my {kind}")
                else:
                    message = prompt.rsplit("Message:", 1)[1]
                    replies.append("lost_card" if "card" in message else "exchange_rate")
            return replies

    teacher = KeywordTeacher()
    gold = LabelledSet(texts=TRAIN.texts[:2] + TRAIN.texts[4:6], labels=[0, 0, 1, 1])
    val = TRAIN
    pool_texts = TRAIN.texts[2:4] + TRAIN.texts[6:]

    examples = pick_examples(gold, labels, per_class=1, seed=0)
    val_pred = classify_with_teacher(teacher, val.texts, labels, examples)
    assert teacher_verdict(accuracy(val.labels, val_pred), 0.5, len(val)).split(":")[0] in {
        "better",
        "unclear",
        "worse",
    }

    pool_labelled, dropped = labelled_from_predictions(
        pool_texts, classify_with_teacher(teacher, pool_texts, labels, examples)
    )
    assert dropped == 0 and len(pool_labelled) == len(pool_texts)

    seeds = {
        label: [t for t, y in zip(gold.texts, gold.labels, strict=True) if y == label]
        for label in (0, 1)
    }
    generated = generate_examples(teacher, labels, n_per_class=3, seed_examples=seeds)
    generated, repeats = dedupe(generated, against=val.texts)
    generated, off_type = round_trip_filter(teacher, generated, labels, examples)
    assert repeats >= 2  # each type's reply repeats a message
    assert off_type == 0

    train = LabelledSet(
        texts=gold.texts + pool_labelled.texts + generated.texts,
        labels=gold.labels + pool_labelled.labels + generated.labels,
    )
    student = train_classifier(tiny_encoder_dir, train, val, num_labels=2, epochs=1, device="cpu")
    predictions = predict(student, val.texts)
    save_result(
        "blackbox_plus_synthetic",
        summarize(val.labels, predictions, 2, n_resamples=50),
        predictions,
        tmp_path,
    )
    assert "blackbox_plus_synthetic" in load_results(tmp_path)


def test_lora_lab_flow(tiny_lm, tmp_path):
    import copy

    pytest.importorskip("peft")
    from kd_course.generative import (
        measure_training,
        predict_generative,
        train_generative_classifier,
    )
    from kd_course.lora import add_lora, count_parameters_by_kind, folder_size_mb, merge_lora

    labels = ["cat", "dog"]
    train = LabelledSet(
        texts=["the cat sat on the mat", "a cat sat", "the dog ran to the floor", "a dog ran"],
        labels=[0, 0, 1, 1],
    )
    base, tokenizer = tiny_lm

    # Run A: full fine-tune
    full = copy.deepcopy(base)
    with measure_training(full) as measured:
        train_generative_classifier(full, tokenizer, train, labels, epochs=1, batch_size=2)
    full_counts = count_parameters_by_kind(full)
    assert measured.cost.seconds > 0

    # Run B: LoRA, saved as an adapter, then merged
    lora_model = add_lora(copy.deepcopy(base), rank=2, alpha=4, target_modules=["c_attn"])
    train_generative_classifier(
        lora_model, tokenizer, train, labels, epochs=1, learning_rate=1e-3, batch_size=2
    )
    lora_counts = count_parameters_by_kind(lora_model)
    assert lora_counts["trainable"] < full_counts["trainable"]

    lora_model.save_pretrained(tmp_path / "adapter")
    full.save_pretrained(tmp_path / "full")
    assert 0 < folder_size_mb(tmp_path / "adapter") < folder_size_mb(tmp_path / "full")

    before = predict_generative(lora_model, tokenizer, train.texts, labels)
    merged = merge_lora(lora_model)
    assert predict_generative(merged, tokenizer, train.texts, labels) == before

    predictions = predict_generative(merged, tokenizer, train.texts, labels)
    save_result(
        "generative_lora",
        summarize(train.labels, predictions, 2, n_resamples=50),
        predictions,
        tmp_path,
    )
    assert "generative_lora" in load_results(tmp_path)


def test_generative_lab_flow(tiny_lm, tmp_path):
    import copy

    from kd_course.generative import generate_replies, train_text_pairs
    from kd_course.teachers import LocalTeacher
    from kd_course.titles import (
        acceptance,
        clean_title,
        generate_titles,
        judge_titles,
        score_titles,
        student_prompt,
        usable_for_training,
    )
    from kd_course.token_kd import topk_targets, train_token_kd

    base, tokenizer = tiny_lm
    messages = ["the cat sat on the mat", "a dog ran to the floor", "the cat ran", "a dog sat"]

    class ScriptedTeacher:
        """Writes a fixed title per message, and says yes to titles containing 'cat' or 'dog'."""

        def complete(self, prompts, max_new_tokens=16):
            replies = []
            for prompt in prompts:
                if "Is this a good title" in prompt:
                    title = prompt.split("Title:")[1]
                    replies.append("yes" if "cat" in title or "dog" in title else "no")
                else:
                    replies.append(
                        "the cat sat"
                        if "cat" in prompt.split("Customer message:")[1]
                        else "a dog ran"
                    )
            return replies

    teacher = ScriptedTeacher()
    titles = generate_titles(teacher, messages)
    keep = [
        i
        for i, (m, t) in enumerate(zip(messages, titles, strict=True))
        if usable_for_training(m, t)
    ]
    assert keep == [0, 1, 2, 3]
    train_messages = [messages[i] for i in keep]
    train_titles = [titles[i] for i in keep]
    prompts = [student_prompt(m) for m in train_messages]

    # Teacher token records (the tiny model stands in for the teacher's weights).
    records = topk_targets(base, tokenizer, prompts, train_titles, k=5, answer_prefix="")

    # In the lab the untrained student gets the teacher's full instructions. The tiny test model
    # can only read 64 tokens, so here it gets the short prompt, through the same wrapper.
    replies = LocalTeacher(base, tokenizer).complete(prompts, max_new_tokens=8)
    untrained = [clean_title(reply) for reply in replies]

    sequence_student = copy.deepcopy(base)
    train_text_pairs(
        sequence_student, tokenizer, prompts, train_titles, epochs=1, batch_size=2, answer_prefix=""
    )
    sequence_titles = [
        clean_title(r) for r in generate_replies(sequence_student, tokenizer, prompts, 8)
    ]

    token_student = copy.deepcopy(base)
    train_token_kd(token_student, tokenizer, prompts, records, epochs=1, batch_size=2)
    token_titles = [clean_title(r) for r in generate_replies(token_student, tokenizer, prompts, 8)]

    folder = tmp_path / "generation"
    for name, outputs in {
        "untrained": untrained,
        "sequence": sequence_titles,
        "token": token_titles,
    }.items():
        scores = score_titles(outputs, titles)
        scores.update(acceptance(judge_titles(teacher, messages, outputs)))
        save_result(name, scores, outputs, folder)
    saved = load_results(folder)
    assert set(saved) == {"untrained", "sequence", "token"}
    assert all(0 <= r["metrics"]["accepted"] <= 1 for r in saved.values())
    # Generation results live in their own folder and do not disturb the classification table.
    assert load_results(tmp_path) == {}


def test_serving_lab_flow(tiny_encoder_dir, tmp_path):  # noqa: F811
    """Save, shrink, serve through the real web app, load-test, and build the fallback table."""
    from fastapi.testclient import TestClient

    from kd_course.serving.app import create_student_app
    from kd_course.serving.cascade import cascade_stats, correct_flags, sweep_thresholds
    from kd_course.serving.export import (
        ClassifierPredictor,
        load_classifier,
        model_size_mb,
        quantize_int8,
        save_classifier,
    )
    from kd_course.serving.loadtest import cost_per_1k, run_load_test, summarize_latencies

    labels = ["lost_card", "exchange_rate"]
    student = train_classifier(
        tiny_encoder_dir,
        TRAIN,
        TRAIN,
        num_labels=2,
        epochs=30,
        learning_rate=5e-3,
        batch_size=4,
        device="cpu",
    )
    folder = save_classifier(student, labels, tmp_path / "students" / "classifier")

    small = quantize_int8(student.model)
    assert model_size_mb(small) < model_size_mb(student.model)
    in_notebook = ClassifierPredictor(student.model, student.tokenizer, student.max_length)
    results = in_notebook.predict(TRAIN.texts)

    predictor, loaded_labels = load_classifier(folder, device="cpu", quantize=True)
    client = TestClient(create_student_app(predictor, loaded_labels))

    def call_service(batch):
        response = client.post("/classify", json={"texts": list(batch)})
        response.raise_for_status()
        return response.json()

    batches = [TRAIN.texts[i : i + 4] for i in range(0, len(TRAIN.texts), 4)]
    load = run_load_test(call_service, batches * 3, concurrency=2)
    assert load.errors == 0 and load.items == 3 * len(TRAIN.texts)
    assert summarize_latencies(load.latencies)["p95_ms"] > 0
    assert cost_per_1k(1.62, load.items_per_second) > 0

    served = call_service(TRAIN.texts)["predictions"]
    assert [p["label"] for p in served] == [labels[label] for label, _ in results]

    confidences = [confidence for _, confidence in results]
    student_right = correct_flags([label for label, _ in results], TRAIN.labels)
    teacher_right = [True] * len(TRAIN.texts)
    rows = sweep_thresholds(confidences, student_right, teacher_right, [0.0, 0.5, 1.01])
    assert rows[-1]["accuracy"] == 1.0 and rows[-1]["sent_to_teacher"] == 1.0
    chosen = next(row["threshold"] for row in rows if row["accuracy"] >= rows[-1]["accuracy"])
    assert cascade_stats(confidences, student_right, teacher_right, chosen)["accuracy"] == 1.0


def test_vision_lab_flow(tiny_clip, tiny_vit_dir, tmp_path):  # noqa: F811
    """Module 8 end to end on tiny models: score with the teacher, train three students, compare."""
    pytest.importorskip("peft")
    import transformers

    from kd_course.label_scoring import top_labels
    from kd_course.lora import count_parameters_by_kind
    from kd_course.vision import (
        make_image_splits,
        predict_images,
        to_pixels,
        train_image_classifier,
        zero_shot_or_load,
    )

    from .test_vision import toy_set

    teacher, teacher_processor, teacher_tokenizer = tiny_clip
    vit_dir = tiny_vit_dir

    labels = ["red_thing", "blue_thing"]
    splits = make_image_splits(
        toy_set(16), toy_set(8, offset=200), labels_per_class=3, val_per_class=2, test_size=8
    )
    scores = {
        name: zero_shot_or_load(
            tmp_path / "teacher_scores" / f"{name}.json",
            part,
            teacher,
            teacher_processor,
            teacher_tokenizer,
            labels,
            template="a photo of a {}",
        )
        for name, part in (
            ("test", splits.test),
            ("gold_train", splits.gold_train),
            ("pool", splits.pool),
        )
    }
    teacher_pred = top_labels(scores["test"])
    folder = tmp_path / "vision"
    save_result(
        "teacher_zero_shot",
        summarize(splits.test.labels, teacher_pred, 2, n_resamples=50),
        teacher_pred,
        folder,
    )

    processor = transformers.AutoImageProcessor.from_pretrained(vit_dir)
    gold, val = (
        to_pixels(processor, splits.gold_train.images),
        to_pixels(processor, splits.gold_val.images),
    )
    pool, test = to_pixels(processor, splits.pool.images), to_pixels(processor, splits.test.images)

    common = {
        "val_pixels": val,
        "val_labels": splits.gold_val.labels,
        "num_labels": 2,
        "device": "cpu",
    }
    baseline = train_image_classifier(vit_dir, gold, splits.gold_train.labels, epochs=1, **common)

    import torch

    both = torch.cat([gold, pool])
    mixed_labels = splits.gold_train.labels + [None] * len(splits.pool)
    teacher_scores = scores["gold_train"] + scores["pool"]
    distilled = train_image_classifier(
        vit_dir, both, mixed_labels, teacher_log_probs=teacher_scores, epochs=1, **common
    )
    lora = train_image_classifier(
        vit_dir,
        both,
        mixed_labels,
        teacher_log_probs=teacher_scores,
        epochs=1,
        lora_rank=2,
        **common,
    )
    assert count_parameters_by_kind(lora.model)["trainable_share"] < 1.0

    for name, student in (
        ("baseline", baseline),
        ("distilled", distilled),
        ("distilled_lora", lora),
    ):
        predictions = predict_images(student, test)
        save_result(
            name, summarize(splits.test.labels, predictions, 2, n_resamples=50), predictions, folder
        )
    assert set(load_results(folder)) == {
        "teacher_zero_shot",
        "baseline",
        "distilled",
        "distilled_lora",
    }
    assert "distilled_lora" in format_results(load_results(folder))


def test_embedding_lab_flow(tiny_encoder_dir):  # noqa: F811
    """Module 9 end to end on a tiny model: embed, distil, score, store, search, index."""
    import copy

    import torch
    import transformers

    from kd_course.embeddings import (
        embed_texts,
        neighbour_overlap,
        per_query_precision,
        retrieval_scores,
        top_neighbours,
        train_embedding_student,
    )
    from kd_course.metrics import paired_mean_difference
    from kd_course.vector_store import open_store

    teacher = transformers.AutoModel.from_pretrained(tiny_encoder_dir).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(tiny_encoder_dir)
    corpus, corpus_labels = TRAIN.texts, TRAIN.labels
    queries, query_labels = ["my card stolen", "the exchange rate"], [0, 1]

    teacher_corpus = embed_texts(teacher, tokenizer, corpus)
    teacher_found = top_neighbours(embed_texts(teacher, tokenizer, queries), teacher_corpus, k=3)

    torch.manual_seed(5)
    student = copy.deepcopy(teacher)
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.2)

    def student_results():
        found = top_neighbours(
            embed_texts(student, tokenizer, queries), embed_texts(student, tokenizer, corpus), k=3
        )
        return found, per_query_precision(found, query_labels, corpus_labels)

    found_before, precision_before = student_results()
    train_embedding_student(
        student,
        tokenizer,
        corpus,
        teacher_corpus,
        epochs=20,
        learning_rate=2e-3,
        batch_size=8,
        device="cpu",
    )
    found_after, precision_after = student_results()

    assert 0 <= neighbour_overlap(found_before, teacher_found) <= 1
    assert neighbour_overlap(found_after, teacher_found) >= neighbour_overlap(
        found_before, teacher_found
    )
    assert retrieval_scores(found_after, query_labels, corpus_labels)["k"] == 3
    change = paired_mean_difference(precision_before, precision_after, n_resamples=50)
    assert change.low <= change.difference <= change.high

    # Store the student's vectors and search them, with whichever store is configured.
    vectors = embed_texts(student, tokenizer, corpus)
    store = open_store(dim=vectors.shape[1], table="kd_test_pipeline")
    store.create()
    store.add(list(range(len(corpus))), corpus, vectors.tolist(), corpus_labels)
    query_vector = embed_texts(student, tokenizer, queries[:1])[0].tolist()
    exact = [hit.id for hit in store.search(query_vector, k=3)]
    assert exact == found_after[0]
    store.create_index()
    indexed = [hit.id for hit in store.search(query_vector, k=3)]
    assert neighbour_overlap([exact], [indexed]) == 1.0
    assert store.size_bytes() > 0
    store.drop()
