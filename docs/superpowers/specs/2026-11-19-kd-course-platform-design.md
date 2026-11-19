# Knowledge Distillation Course: Platform and Module 1 Design

Date: 2026-11-19
Status: awaiting approval

## Intent

Build a self-paced course that teaches new data scientists to take what a
state-of-the-art "teacher" model can do on one specific task and move it into
a small transformer "student". The course starts with text, later covers
images and other data types, and includes building and using LoRA adapters.

Students read lessons on a hosted website and run labs in notebooks on a GPU
they reach over SSH. The same site must be easy for anyone to stand up
elsewhere from the repo.

By the end, a student can take a task from their job and build the full
pipeline: set baselines, choose black-box or white-box distillation, train
with full fine-tuning or LoRA, evaluate, and serve the student model.

## Goals

- Lessons are served as a web app, not READMEs, in the "Studio" visual
  direction chosen from the mockups (bold display type, generous white space,
  deep teal with a warm accent, light and dark themes).
- Each concept is taught with the learning mode that suits it: interactive
  simulations, predict-then-observe questions, annotated worked examples,
  retrieval quizzes with explanatory feedback, spaced review, concept maps,
  and hands-on labs.
- The course keeps three ideas clearly apart: training on teacher-generated
  data (black-box), learning from a teacher's softened probabilities
  (white-box), and LoRA as a way of updating weights that works with either.
- Every lab runs on one A10G (24 GB) in under an hour.
- Instructor tracking is a switch: on for the purpose-built instance, off by
  default everywhere else.
- Standing the site up elsewhere takes a clone and a few commands, with no
  containers.

## Non-goals

- Containers (Docker, Compose). They are hard to use in the target environment.
- Grades, certificates, or LMS integration.
- Hosting notebooks or GPUs from the site (no JupyterHub).
- Video content.
- Image and multimodal modules in this first project. They come later.
- Support for V100 / P3 hardware.

## Anti-goals

- An external deployment that tracks students without the operator opting in.
- A student needing Node or a database just to read lessons on the hosted site.
- Labs that need more than one A10G.
- A site that looks like a stock documentation theme.

## Constraints

- Site host: a CPU instance. Lab host: g5.4xlarge (1x A10G 24 GB, 16 vCPU,
  64 GB RAM), reached over SSH.
- No containers.
- Python for all backend and lab code; FastAPI, SQLAlchemy and Postgres where
  a backend and database are needed.
- Work lands in small increments: platform plus Module 1 first, then one
  module per cycle.

## Approaches considered

1. **FastAPI + Jinja + Markdown lessons.** One Python toolchain, full control
   of the look. Recommended by the designer for simplicity.
2. **Docs generator (MkDocs Material / Quarto).** Fastest to build, but looks
   like a docs site and makes custom interactives awkward. Rejected: cannot
   meet the visual goal.
3. **Astro + MDX with interactive components.** Richest component model for
   interactive lessons; adds a Node build step. **Chosen by the course owner.**

To keep option 3 from becoming two services to run, the Astro site is built
to static files and the FastAPI app serves those files alongside its API.
Node is needed only to build; one Python process runs in production.

## Design

### Repository layout

```
site/            Astro project: lessons (MDX), components, design tokens
server/          FastAPI app: serves built site, config and tracking API
src/kd_course/   Python package shared by labs (data, losses, evaluation)
labs/            Jupyter notebooks, one folder per module
tests/           pytest for server and kd_course
docs/            Design docs and operator guide
```

One `pyproject.toml` managed with uv. Dependency groups keep installs small:
`server` (FastAPI, SQLAlchemy) for the CPU instance, `labs` (torch,
transformers, peft) for the GPU instance.

### Site (Astro)

- Lessons are MDX files in an Astro content collection, one folder per
  module. Front matter carries title, order, estimated time, objectives, and
  the lab path. The sidebar and progress bar are generated from it.
- Interactive parts are components used inside MDX: `Quiz`, `PredictReveal`,
  `ReviewCard`, `ConceptGrid`, `WorkedExample`, `LabCard`, plus per-concept
  simulations (for example `TemperatureExplorer` in Module 3).
- Design tokens (colors, type, spacing, radius) live in one CSS file with
  light and dark values. Fonts are bundled with the site, not loaded from a
  third party, so it works on restricted networks.
- Accessibility: full keyboard use, visible focus, reduced-motion support,
  WCAG AA contrast in both themes.

### Server (FastAPI)

- Serves the built site from `site/dist` and exposes `/api`.
- `GET /api/config` tells the front end whether tracking is on.
- Settings come from environment variables (pydantic-settings).
  `TRACKING_ENABLED` defaults to `false`.

### Progress and tracking

The front end writes progress through one small interface with two
implementations, chosen from `/api/config`:

- **Tracking off (default).** Progress and quiz answers are saved in the
  browser. No accounts, no database. The server never connects to Postgres.
- **Tracking on.** There is no login page. Every request carries a JWT
  issued by the hosting environment that identifies the user. The server
  verifies the token's signature, expiry, issuer and audience, then reads the
  user's identity from its claims. An unverified or missing token is
  rejected; claims are never trusted without verification. Progress and quiz
  attempts are posted to the API and stored in Postgres (SQLAlchemy 2,
  Alembic migrations). An instructor page shows completion per student and
  per lesson, and quiz results per question. The site shows students a notice
  that progress is visible to the instructor.

  The token arrives in a request header added by the hosting environment.
  The header name is a setting (`AUTH_HEADER_NAME`); a `Bearer ` prefix is
  accepted if present.

  The exact claims and signing keys are not known yet, so all of these are
  settings filled in at deployment: issuer, audience, public key or JWKS
  address, algorithm, and the claim names for user ID, name and email. Only
  asymmetric algorithms from an explicit allow-list are accepted. With
  tracking on, the server refuses to start if the verification settings are
  missing, so it can never run in a mode that trusts unverified tokens.

  Instructors are identified by a configurable role or group claim
  (`INSTRUCTOR_CLAIM`, `INSTRUCTOR_CLAIM_VALUE`), with a fallback allow-list
  of user IDs (`INSTRUCTOR_IDS`) in case the token carries no role.

  Tests sign their own tokens with a throwaway key pair.

Tables: `students`, `lesson_progress`, `quiz_attempts`.

### Labs

- Notebooks live in `labs/`. Lesson pages show the notebook path and the
  command to open it. A student can save their Jupyter address once in the
  site settings so lab buttons link straight to it.
- Shared code lives in `kd_course` with tests, so notebooks stay short.
- Model defaults: teacher and student from one family so they share a
  tokenizer (Qwen2.5-7B-Instruct with Qwen2.5-0.5B / 1.5B); ModernBERT for
  classification students. Teacher probabilities are computed offline and
  cached as top-k, so teacher and student never share the GPU.

### Course outline

1. Foundations
2. Baselines and evaluation
3. White-box distillation (soft probabilities, temperature, KD loss)
4. Black-box distillation (teacher-generated data, filtering, terms of service)
5. LoRA and QLoRA
6. Distilling generative models
7. Compress and serve (quantize, merge adapters, FastAPI, cost comparison)
8. Vision and multimodal (later project)

Optional: embedding distillation with pgvector search.

### First project scope: platform plus Module 1

Platform: Astro shell in the Studio design, core components, FastAPI server,
browser-stored progress, tracking mode with JWT identity and instructor page,
operator guide, CI.

Module 1 lessons:

1. Why distill: cost, latency, and control (with a cost and latency calculator)
2. The map: black-box or white-box, full fine-tune or LoRA (concept grid)
3. Fine-tuning, distillation, and LoRA: what each term means and does not mean
4. Licenses and terms of service
5. Lab: check the GPU, run the teacher and student, measure speed and memory

### Running it

- Elsewhere: `uv sync --group server`, build the site (`npm ci && npm run
  build` in `site/`), start with one `uv run` command. A `Makefile` wraps
  these.
- Purpose-built instance: the same, plus Postgres installed on the host, a
  systemd unit, and `TRACKING_ENABLED=true`. Documented in the operator guide.

### Testing

- pytest for the server (both tracking modes) and `kd_course`.
- Vitest for component logic (quiz scoring, progress store, simulations).
- `astro check` and a build in CI (GitHub Actions).
- One Playwright smoke test: open a lesson, answer a quiz, see progress saved.

## Designer's calls (not yet confirmed by the course owner)

- FastAPI serves the built Astro site, so production is one Python process.
- Interactive components are written in React (largest ecosystem and
  documentation).
- Students are told on screen when tracking is on.
- Running case study: support-request triage, starting from the Banking77
  intent dataset for classification. To be confirmed in the Module 2 design.
- Embedding distillation is an optional module after Module 7.

## Left to the builder

- Exact component props and MDX authoring conventions.
- API route shapes and table columns beyond those named above.
- Token values, starting from the Studio mockup.
- Lesson copy and the Module 1 notebook contents.
