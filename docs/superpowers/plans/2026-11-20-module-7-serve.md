# Module 7: Compress and Serve — Design Note

**Spec:** `docs/superpowers/specs/2026-11-19-kd-course-platform-design.md`

## Decisions

- **The capstone serves the encoder student from Module 3.** For classification it does the same job as the generative students at a fraction of the running cost, which lesson 7.1 uses to teach how to choose what to serve.
- **A second, separate FastAPI app** (`kd_course.serving.app`, command `kd-student-serve`) serves the student. It is not part of the course site. Endpoints: `/health`, `/info`, `/classify`. Requests are validated by type (1 to 64 texts, each 1 to 2,000 characters); the model is loaded once; the handler is a plain `def` so it runs in a worker thread; one request at a time reaches the model; failures return a fixed message and log the cause.
- **No authentication** in the service. It binds to 127.0.0.1 by default and the docs say to keep it there unless something in front provides authentication.
- **Quantization is post-training dynamic 8-bit on CPU**, the simplest kind and one that can be tested without a GPU. 4-bit GPU loading stays in Module 5.
- **Cost is computed from measured throughput** and an hourly price the student sets; the lessons say the figure assumes a busy machine and leaves out people.
- **Fallback to the teacher:** the confidence threshold is chosen on validation data (the lab scores the validation messages with the teacher for this) and the cascade is scored once on the test set.
- **Lab 7 retrains and saves the student**, because Lab 3 did not save its model.

## Lessons

7.1 From a trained model to a service · 7.2 Fewer bits per weight (quantization explorer) · 7.3 Serving with FastAPI (annotated code) · 7.4 Speed and cost · 7.5 The teacher as a fallback (cascade explorer) · 7.6 Lab

## Verified here

- The real `kd-student-serve` command was started on CPU with a small saved 8-bit model and answered `/info` and `/classify`, rejected an empty request with 422, and exited with a clear message for a missing folder.
- An end-to-end test saves a student, quantizes it, serves it through the real app, load-tests it and builds the fallback table.

## Known limits

- Not run on a GPU. Dynamic 8-bit quantization of ModernBERT on the students' machines is untested; the lab says to skip that section if it fails.
- Throughput and cost figures depend on hardware and are not predicted in the lessons.
- The cascade explorer in lesson 7.5 uses made-up data, labelled as such.
