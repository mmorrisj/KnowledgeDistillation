# Module 5: LoRA and QLoRA — Design Note

**Spec:** `docs/superpowers/specs/2026-11-19-kd-course-platform-design.md`

## Decisions

- **The student becomes a text-generation model** (Qwen2.5-0.5B and 1.5B, both Apache 2.0), trained to write the type name. The task and test set stay the same, so results line up with Modules 2 to 4.
- **Short student prompt:** `Message: ...\nType:` with no list of types. A trained student knows the types; the prompt drops from about 600 tokens to about 30. This keeps every run to a minute or two and is itself a lesson about serving cost.
- **Training data is held fixed** across runs: gold labels plus the pool labelled with the teacher's top choice from the Lab 3 scores. Only the update method and student size change. This is the "teacher-written labels with LoRA" cell of the course map.
- **LoRA is taught twice:** a from-scratch `LoRALinear` (tested: identical at start, only A and B train, merge is exact) for understanding, and the PEFT library through `add_lora` for the lab.
- **Precision:** full fine-tuning loads 32-bit weights and uses bf16 autocast; LoRA keeps the frozen base in bf16 and lets PEFT hold the adapters in 32-bit.
- **Rank comparison uses validation data only**; test scores for those runs are not computed.
- **QLoRA** is taught in lesson 5.4 and offered as an optional final lab cell on the 7B model.

## Lessons

5.1 The cost of moving every weight (memory explorer) · 5.2 How LoRA works (size explorer, from-scratch code) · 5.3 Choosing the settings (sorting drill) · 5.4 QLoRA and merging · 5.5 Lab

## Known limits

- Not run on a GPU. Times, memory figures and the bf16/autocast path are unverified on real hardware.
- The QLoRA cell and `load_causal_lm_4bit` cannot be tested without a GPU and are marked optional.
- Memory figures in the memory explorer are rules of thumb (16, 2 and 0.7 bytes per weight).
