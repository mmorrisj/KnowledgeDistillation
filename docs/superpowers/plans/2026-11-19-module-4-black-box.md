# Module 4: Black-box Distillation — Design Note

**Spec:** `docs/superpowers/specs/2026-11-19-kd-course-platform-design.md`

## Decisions

- **Written to cover both outcomes** of the unmeasured Lab 2 result (course owner's instruction). The module opens with a check: on validation data, is the teacher clearly better than the baseline, clearly worse, or too close to call? `teacher_verdict` prints which, and lessons 4.1 and 4.5 say what each case means. The lab runs to the end in every case.
- **Teacher abstraction:** `Teacher.complete(prompts) -> replies`. `LocalTeacher` wraps the local Qwen model so only text goes in and out. `ApiTeacher` speaks the OpenAI-style chat completions protocol, counts tokens, retries rate limits and server errors, and never includes the key or response body in error messages. The lab uses the local teacher so it needs no account.
- **Improving the teacher first:** plain prompt against one example per type, compared on validation. Examples come from the gold training set only.
- **Labelling:** unusable replies are dropped from training data (and counted). The lab labels 3,000 pool messages by default to keep within time.
- **Generation:** one request per type, seeded with two gold training messages; duplicates removed, including anything matching test or validation messages; round-trip check drops off-type messages.
- **Lab 2 now saves the baseline's validation accuracy**, which the verdict needs.

## Lessons

4.1 Learning from answers alone · 4.2 A better teacher from the same model (prompt cost estimator) · 4.3 Labelling the pool · 4.4 Writing new examples (sorting drill) · 4.5 Lab

## Known limits

- Not run on a GPU. Times are estimates.
- `ApiTeacher` is tested against a simulated server, not a real provider.
- Requests to an API teacher are sent one at a time; a large pool would benefit from concurrency, which is left for later.
