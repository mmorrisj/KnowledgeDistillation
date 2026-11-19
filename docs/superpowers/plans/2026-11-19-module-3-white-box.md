# Module 3: White-box Distillation — Design Note

**Spec:** `docs/superpowers/specs/2026-11-19-kd-course-platform-design.md`

## Decisions

- **Teacher signal:** the Qwen2.5-7B teacher's probability for each of the 77 types, read by scoring how likely it is to write each type name after the Module 2 prompt. No text is generated.
- **Speed:** the shared part of the prompt (instructions and type list) is read once and its internal state reused for every message; all 77 type names are scored in one batch per message. Verified against a one-pass-per-label calculation on a tiny CPU model.
- **Scores are saved** under `results/teacher_scores/` and tagged with the texts they belong to, so the teacher runs once and never shares the GPU with the student.
- **Student:** ModernBERT-base, as in Module 2. Teacher and student tokenizers differ, which is fine because the probabilities are over types, not tokens. Earlier lesson wording that implied otherwise was corrected.
- **Loss:** `alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * CE(human labels)`. Pool messages have no human label and take part only in the teacher term.
- **Comparison built into the lab:** a student trained on the teacher's top answers against one trained on its probabilities, on the same messages, so the value of the full distribution is measured directly.
- **Temperature is chosen on the validation set**, tying back to Module 2.
- **Results folder** is now shared across labs: `results/` at the top of the repo, or `KD_COURSE_RESULTS_DIR`.

## Lessons

3.1 What the teacher knows · 3.2 Reading a teacher's probabilities · 3.3 Temperature (explorer) · 3.4 The distillation loss (explorer, worked example) · 3.5 Lab

## Known limits

- Not run on a GPU. Scoring time for about 11,000 messages on an A10G is an estimate.
- Whether probabilities beat top answers on this task is unknown until a real run; the lab lesson treats either outcome as a valid finding.
- Hyperparameters (5 epochs, batch 32, T = 2, alpha = 0.5) are starting values.
