# Module 2: Baselines and Evaluation — Design Note and Plan

**Spec:** `docs/superpowers/specs/2026-11-19-kd-course-platform-design.md`

## Decisions

- **Case study:** Banking77 (`PolyAI/banking77`, CC BY 4.0): 10,003 training and 3,080 test messages in 77 types.
- **Scarce-label scenario (chosen by the course owner).** Students pretend they have only 8 human labels per type for training and 4 per type for validation. The other training messages form an unlabelled pool that the teacher labels in later modules. A reference run on all labels shows what plenty of labels would buy.
- **Fixed splits, seed 13:** `gold_train` 616, `gold_val` 308, `pool` 9,079 (true labels kept aside for later analysis of teacher mistakes, never used for training), `test` 1,000 drawn from the official test split.
- **Three baselines, all scored on the same 1,000 test messages:** the prompted teacher (Qwen2.5-7B-Instruct, no training), the student trained on the 616 gold labels (ModernBERT-base), and the full-label reference.
- **Metrics:** accuracy and macro-F1, each with a bootstrap 95% interval; paired bootstrap for differences; "gap closed" = (distilled − baseline) / (teacher − baseline). A teacher reply that names no valid type counts as wrong.
- **Results are saved** to `results/<name>.json` so later modules can compare against them.
- **Code style:** a plain PyTorch training loop, not a trainer framework, so students can read every step.

## Lessons

| Lesson | Teaches | Interactive part |
|---|---|---|
| 2.1 Three numbers before you distill | Ceiling, baseline, distilled; gap closed; the scarce-label scenario and why | Gap-closed calculator |
| 2.2 Splits and leakage | Train, validation, test, pool; how the test set leaks, including through the teacher | "Leak or fine?" sorting drill |
| 2.3 Choosing a metric | Accuracy, macro-F1, per-type scores; unparseable answers; cost and speed as metrics | Imbalance explorer |
| 2.4 Is the difference real? | Sampling error, confidence intervals, test-set size, paired comparison | Interval explorer |
| 2.5 Lab: baselines | Produce the three baselines and their intervals | Notebook |

## Tasks

1. `kd_course.data`: `load_banking77`, `make_splits` (tests on small synthetic data).
2. `kd_course.metrics`: `accuracy`, `macro_f1`, `per_class_f1`, `bootstrap_ci`, `paired_bootstrap`, `gap_closed`.
3. `kd_course.prompted`: `build_classification_prompt`, `parse_label`, `classify_with_llm`.
4. `kd_course.classifier` (`train_classifier`, `predict`) and `kd_course.results` (`save_result`, `load_results`, `format_results`).
5. Site: `stats.ts`, `GapClosedCalculator`, `ImbalanceExplorer`, `IntervalExplorer`.
6. Lessons 2.1 to 2.4 and browser tests.
7. Lab notebook, lesson 2.5, CI and docs.

**Known limit:** the lab cannot be executed in the build environment (no GPU). Package code is tested on CPU with tiny models. Expected scores quoted in lessons are described as rough expectations, not measurements.
