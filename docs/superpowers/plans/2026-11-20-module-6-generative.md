# Module 6: Distilling Generative Models — Design Note

**Spec:** `docs/superpowers/specs/2026-11-19-kd-course-platform-design.md`

## Decisions

- **New task on the same data:** write a ticket title of at most eight words for each Banking77 message. It has many valid outputs, a rule a program can check, and short answers that keep the lab fast.
- **No human references exist**, so the baseline is the untrained 0.5B student given the teacher's full instructions.
- **Two methods, mirroring Modules 4 and 3:** sequence-level (train on the teacher's titles) and token-level (also match the teacher's top-20 next-token probabilities at every title position). Token-level uses forward KL on saved teacher records.
- **Teacher records are read in one pass** over prompt plus title, without generating. Checked against a direct calculation on a tiny model.
- **Different prompts, same answer tokens.** The teacher keeps its full instruction prompt in chat format; the student uses a short prompt. Only the title's tokens must line up, which the shared Qwen tokenizer guarantees; the lab asserts the vocabularies match.
- **Rule filtering before training:** titles over the limit, empty, or copying the message are dropped, so students can follow the rule more reliably than the teacher.
- **Three kinds of score, reported together:** rule compliance, word overlap with the teacher (named fidelity, not quality), and yes/no acceptance by a judge. The judge is the teacher, and lessons 6.5 and 6.6 say plainly what bias that brings.
- **Reverse KL and learning from the student's own outputs** are taught as concepts in lesson 6.4 and not implemented.
- **Generation results are saved in `results/generation/`** so they do not mix with the classification table.

## Lessons

6.1 From choosing to writing · 6.2 Learning from the teacher's text · 6.3 Matching every token (top-k storage explorer) · 6.4 Two directions of matching (KL explorer) · 6.5 Scoring generated text (sorting drill) · 6.6 Lab

## Known limits

- Not run on a GPU. Whether token-level beats sequence-level on this short task is unknown; the lab lesson treats a small or unclear gain as a normal result.
- The judge shares the teacher's preferences. A second, different judge model would be better and is not included.
- Students are fully fine-tuned; combining token-level distillation with LoRA is described but not exercised.
