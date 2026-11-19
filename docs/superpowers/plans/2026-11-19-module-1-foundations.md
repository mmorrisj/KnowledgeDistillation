# Module 1: Foundations Implementation Plan (Plan 3 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The five Foundations lessons, the setup lab and the first slice of the shared `kd_course` Python package, so a student can read Module 1 and verify their GPU machine is ready.

**Architecture:** Lessons are MDX files using the existing components plus four new ones (cost calculator, training-loop explorer, sorting drill, pipeline diagram). The lab is a short notebook that calls tested functions in `src/kd_course`; all logic lives in the package, none in the notebook. Package tests run on CPU with a tiny randomly initialized model, so CI needs no GPU and no downloads.

**Tech Stack:** As Plan 1, plus PyTorch, Transformers, nbformat (Python `labs` group).

**Spec:** `docs/superpowers/specs/2026-11-19-kd-course-platform-design.md`

## Global Constraints

- Every lab runs on one A10G (24 GB) in under an hour. This lab targets 5 to 10 minutes after models are downloaded.
- Model defaults: teacher `Qwen/Qwen2.5-7B-Instruct`, student `Qwen/Qwen2.5-0.5B-Instruct`.
- `uv sync` on the site host must not install PyTorch: lab dependencies live only in the `labs` group, which is not a default group.
- Notebooks are committed with outputs cleared.
- Lesson copy: plain language for new data scientists; no claim about a specific provider's prices or terms as current fact. Prices in the calculator are editable example values, labelled as examples.
- Lessons use `client:load` on interactive components.
- The component sampler stays buildable for authors but is not shown to students in the sidebar or on the home page.

## Review Focus

1. The lab machine has no GPU, or a GPU without bf16 (for example a V100): the check reports a clear problem line, it does not crash. (Task 1)
2. Models are pre-downloaded to a local folder because the GPU machine cannot reach Hugging Face: the lab loads from that folder. (Task 2)
3. The calculator gets zero, negative or empty inputs: it shows a message, never `NaN` or `Infinity`. (Task 3)
4. A student answers a sorting-drill item wrong, then right: the score counts the item once and the explanation shows both times. (Task 4)
5. A lesson links to a later module that is not written yet: the link text says it is coming, and there is no dead link. (Task 5)

---

### Task 1: `kd_course` package and environment check

**Files:**
- Create: `src/kd_course/__init__.py`, `src/kd_course/env.py`, `tests/kd_course/test_env.py`
- Modify: `pyproject.toml` (wheel packages `server`, `src/kd_course`; group `labs`: `torch`, `transformers`, `accelerate`, `nbformat`, `ipykernel`; `dev` adds `nbformat`)

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class GpuInfo: name: str; memory_gb: float; supports_bf16: bool
@dataclass(frozen=True)
class EnvReport:
    python: str; torch: str | None; gpu: GpuInfo | None; free_disk_gb: float; problems: list[str]
    @property
    def ok(self) -> bool            # no problems
    def summary(self) -> str        # multi-line, human readable
def check_environment(min_gpu_gb: float = 22.0, min_disk_gb: float = 40.0,
                      probe: Callable[[], GpuInfo | None] | None = None,
                      disk_path: str = ".") -> EnvReport
```
- `torch` is imported lazily inside the default probe, so the module imports without it.

- [ ] **Step 1: Write the failing tests**

```python
def test_ok_on_a10g():            # probe → GpuInfo("NVIDIA A10G", 22.5, True) → report.ok, problems == []
def test_no_gpu_is_a_problem():   # probe → None → "No GPU" in problems[0]; not ok
def test_small_gpu_is_a_problem():# 15.8 GB → problem names "22" and the found size
def test_no_bf16_is_a_problem():  # V100-like → problem mentions bf16
def test_low_disk_is_a_problem(): # min_disk_gb=10**6 → problem mentions disk
def test_summary_lists_each_problem_on_its_own_line()
def test_module_imports_without_torch()   # sys.modules["torch"] = None; importlib.reload(env) works
```

- [ ] **Step 2: Run** `uv run pytest tests/kd_course -q`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q && uv run ruff check .`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(kd_course): environment check`

### Task 2: Model loading and generation benchmark

**Files:**
- Create: `src/kd_course/models.py`, `src/kd_course/bench.py`, `tests/kd_course/test_models.py`, `tests/kd_course/test_bench.py`, `tests/kd_course/conftest.py` (fixture `tiny_lm`: a 2-layer GPT-2 built from config with a whitespace `tokenizers` vocabulary, no download)

**Interfaces:**
- Produces:

```python
TEACHER_ID = "Qwen/Qwen2.5-7B-Instruct"; STUDENT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
def resolve_model_path(model_id: str, model_dir: str | None = None) -> str
    # model_dir defaults to env KD_COURSE_MODEL_DIR; returns "<model_dir>/<last path part of model_id>"
    # if that folder exists, else model_id unchanged
def load_causal_lm(model_id: str, device: str = "cuda", dtype: str = "bfloat16") -> tuple[PreTrainedModel, PreTrainedTokenizerBase]
def count_parameters(model) -> int

@dataclass(frozen=True)
class BenchResult: n_params: int; tokens_per_second: float; latency_p50_s: float; peak_memory_gb: float | None; new_tokens: int
def benchmark_generation(model, tokenizer, prompts: list[str], max_new_tokens: int = 64) -> BenchResult
def compare(teacher: BenchResult, student: BenchResult) -> dict[str, float]   # keys: size_ratio, speedup, memory_ratio (omitted if either peak is None)
```
- `peak_memory_gb` is `None` off-GPU. `benchmark_generation` uses greedy decoding and raises `ValueError` on an empty prompt list.

- [ ] **Step 1: Write the failing tests**

```python
def test_resolve_prefers_local_folder(tmp_path)     # tmp/Qwen2.5-0.5B-Instruct exists → that path
def test_resolve_falls_back_to_hub_id(tmp_path)
def test_resolve_reads_env_var(monkeypatch, tmp_path)
def test_count_parameters_matches_manual_sum(tiny_lm)
def test_benchmark_reports_positive_speed_and_token_count(tiny_lm)  # new_tokens == len(prompts) * max_new_tokens upper bound, > 0
def test_benchmark_peak_memory_is_none_on_cpu(tiny_lm)
def test_benchmark_rejects_empty_prompts(tiny_lm)
def test_compare_ratios()   # teacher(n=70, tps=10, mem=14) vs student(n=5, tps=100, mem=1) → 14.0, 10.0, 14.0
def test_compare_omits_memory_when_unknown()
```

- [ ] **Step 2: Run** `uv run --group labs pytest tests/kd_course -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** Tests that need torch are skipped with `pytest.importorskip("torch")`, so the default `uv run pytest` still passes on the site host; CI gets a `labs` job that installs the CPU build of torch and runs them.
- [ ] **Step 4: Run** `uv run --group labs pytest -q`. Expected: PASS with no skips.
- [ ] **Step 5: Commit** `feat(kd_course): model loading and generation benchmark`

### Task 3: Cost calculator

**Files:**
- Create: `site/src/lib/cost.ts`, `site/src/lib/cost.test.ts`, `site/src/components/CostCalculator.tsx`, `site/src/components/CostCalculator.test.tsx`

**Interfaces:**
- Produces:

```ts
export function apiCostPer1k(inTokens: number, outTokens: number, usdPerMTokIn: number, usdPerMTokOut: number): number
export function selfHostedCostPer1k(instanceUsdPerHour: number, requestsPerSecond: number): number
export function breakEvenRequestsPerDay(apiPer1k: number, instanceUsdPerHour: number): number   // requests/day where a machine left on all day costs the same
export function isValidInput(...values: number[]): boolean   // all finite and > 0
CostCalculator: { defaults?: Partial<{ inTokens; outTokens; usdPerMTokIn; usdPerMTokOut; instanceUsdPerHour; requestsPerSecond; requestsPerDay }> }
```

- [ ] **Step 1: Write the failing tests**

```ts
apiCostPer1k(500, 100, 3, 15) ≈ 3.0          // (500*3 + 100*15)/1e6 * 1000
selfHostedCostPer1k(1.6, 20) ≈ 0.0222        // 1.6/3600/20*1000
breakEvenRequestsPerDay(3.0, 1.6) === 12800  // 1.6*24 / (3.0/1000)
isValidInput(1, 0) === false; isValidInput(1, NaN) === false; isValidInput(1, -2) === false
it("shows both costs and which is cheaper at the entered volume")
it("shows a message and no numbers when an input is empty, zero or negative")
it("labels the default prices as examples")
```

- [ ] **Step 2: Run** `npm test`. Expected: FAIL.
- [ ] **Step 3: Implement.** Every input has a visible label and unit; results are in an `aria-live` region.
- [ ] **Step 4: Run** `npm test && npm run check`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(site): cost calculator`

### Task 4: Sorting drill, training-loop explorer, pipeline diagram, hidden modules

**Files:**
- Create: `site/src/components/SortDrill.tsx`, `SortDrill.test.tsx`, `site/src/components/TrainingLoopExplorer.tsx`, `TrainingLoopExplorer.test.tsx`, `site/src/components/PipelineDiagram.astro`
- Modify: `site/src/lib/nav.ts`, `nav.test.ts`, `site/src/content/modules.ts`, `site/src/styles/lesson.css` (table styling for prose)

**Interfaces:**
- Produces:

```ts
SortDrill: { lessonId: string; id: string; categories: string[]; items: { text: string; answer: string; why: string }[] }
// Each item shows category buttons. Choosing shows "Correct." or "Not quite. It is <answer>." plus `why`.
// Footer: "<n> of <total> sorted correctly". Each item recorded via recordQuiz(lessonId, `${id}:${index}`, choice, correct).
TrainingLoopExplorer: {}   // fixed content. Shows the training loop as five stages: Input text → Model weights → Predicted probabilities → Target → Update.
// Mode buttons: "Training from scratch", "Pretraining", "Fine-tuning", "Distillation", "LoRA". Each mode marks which stages change
// (data-changed="true" plus a visible "changes here" tag) and shows a two-sentence explanation.
// Fine-tuning changes Input (your labelled data) ; Distillation changes Target (teacher's probabilities or answers); LoRA changes Update (only small added matrices).
ModuleInfo gains `hidden?: boolean`; buildNav omits hidden modules.
PipelineDiagram: { steps: { label: string; detail: string }[] }   // inline SVG-free, CSS flex with arrows; reads as an ordered list to screen readers
```

- [ ] **Step 1: Write the failing tests**

```ts
it("buildNav omits modules marked hidden")
it("shows the explanation and correct category after a wrong choice")
it("counts an item once when answered wrong then right")     // footer reads "1 of 2 sorted correctly"
it("records each item under its own quiz id")
it("TrainingLoopExplorer: Distillation marks only the Target stage as changed")
it("TrainingLoopExplorer: LoRA marks only the Update stage as changed")
it("TrainingLoopExplorer: every mode shows an explanation, and the first mode is selected at rest")
```

- [ ] **Step 2: Run** `npm test`. Expected: FAIL.
- [ ] **Step 3: Implement.** Mark `00-sample` hidden; its page still builds.
- [ ] **Step 4: Run** `npm test && npm run check && npm run build`. Expected: PASS; `dist/modules/00-sample/01-components/index.html` still exists.
- [ ] **Step 5: Commit** `feat(site): sorting drill, pipeline diagram, hidden modules`

### Task 5: Lessons 1.1 to 1.5

**Files:**
- Create: `site/src/content/lessons/01-foundations/01-why-distill.mdx`, `02-the-map.mdx`, `03-three-ideas.mdx`, `04-licenses-and-terms.mdx`
- Modify: `site/src/content/modules.ts` (`01-foundations` → `available`), `site/e2e/smoke.spec.ts`

**Content each lesson must carry** (the builder writes the prose):

| Lesson | Teaches | Components |
|---|---|---|
| 1.1 Why distill | A large model is a ceiling on quality but costly per request, slow, and outside your control. A small task-specific model can match it on one narrow task. When distilling is not worth it (low volume, task still changing). | PredictReveal (guess the break-even volume), CostCalculator, Quiz |
| 1.2 How transformers learn | Added at the course owner's request. Text becomes tokens; the model outputs a score per possible next token; softmax turns scores into probabilities; training nudges weights to raise the probability of the right token (cross-entropy). How this differs from classical models: those are trained from scratch per task on hand-built features, while a transformer is pretrained once on huge unlabelled text and then adapted. Encoder (classify, extract) versus decoder (generate). Then the payoff: fine-tuning, distillation and LoRA each change one part of the same loop. | PredictReveal, next-token probability bars (static figure), comparison table (classical vs pretrained transformer), TrainingLoopExplorer, Quiz |
| 1.3 The map | Teacher and student. The two independent choices: signal (black-box text or white-box probabilities) and update method (full fine-tune or LoRA). The course pipeline: baseline, teacher signal, train, evaluate, serve. | PipelineDiagram, ConceptGrid, Quiz |
| 1.4 Three ideas kept apart | Fine-tuning = training a pretrained model further on labelled data. Distillation = the labels or targets come from a teacher model. LoRA = training small added matrices and freezing the rest. Distillation answers "where do targets come from"; LoRA answers "which weights change". Common mix-ups. | Comparison table, SortDrill (8 statements sorted into the three), ReviewCard (from 1.2), Quiz |
| 1.5 Licenses and terms | API terms often restrict training other models on outputs; read the current terms for your provider and plan before collecting data. Open-weight licenses differ within one model family (check each model card). Dataset licenses. Personal and confidential data sent to an API. A pre-flight checklist. | PredictReveal, checklist, Quiz, ReviewCard (from 1.4) |

- [ ] **Step 1: Write the failing browser tests**

```ts
for each of the five lesson URLs: h1 visible, no serious or critical axe violations (light and dark)
test("the home page starts at Why distill and does not list the component sampler")
test("the cost calculator updates when requests per day changes")
```

- [ ] **Step 2: Run** `make e2e`. Expected: FAIL (404s).
- [ ] **Step 3: Write the five lessons.** Each has 2 to 4 objectives, about 600 to 900 words, and ends with a Quiz. References to later modules are plain text ("covered in Module 3"), not links.
- [ ] **Step 4: Run** `make test && make e2e`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(course): Module 1 lessons 1.1 to 1.5`

### Task 6: Lesson 1.6 and the setup lab

**Files:**
- Create: `site/src/content/lessons/01-foundations/05-lab-setup.mdx`, `labs/01_foundations/01_setup_check.ipynb`, `labs/README.md`, `tests/labs/test_notebooks.py`
- Modify: `.github/workflows/ci.yml` (job `labs`: CPU torch, `uv run --group labs pytest`), `docs/operator-guide.md`, `Makefile` (`install-labs`)

**Interfaces:**
- Consumes: `check_environment`, `load_causal_lm`, `benchmark_generation`, `compare`, `TEACHER_ID`, `STUDENT_ID` from Tasks 1 and 2.

**Notebook cells, in order:** (1) what this lab does; (2) `check_environment()` and print the summary, stop if not ok; (3) load the student, benchmark on three fixed prompts; (4) free it, load the teacher, benchmark; (5) `compare` and a small table; (6) three questions for the student to answer in a markdown cell (how many times smaller, how much faster, what the student got wrong in its outputs).

- [ ] **Step 1: Write the failing tests** (`tests/labs/test_notebooks.py`, parametrized over every `labs/**/*.ipynb`)

```python
def test_notebook_is_valid(path)                 # nbformat.validate
def test_notebook_has_no_outputs(path)           # every code cell: outputs == [] and execution_count is None
def test_code_cells_parse(path)                  # ast.parse each code cell (IPython magics lines stripped)
def test_imports_from_kd_course_exist(path)      # every `from kd_course.x import a, b` resolves
def test_at_least_one_notebook_found()
```

- [ ] **Step 2: Run** `uv run pytest tests/labs -q`. Expected: FAIL (no notebooks).
- [ ] **Step 3: Write** the notebook, `labs/README.md` (install with `make install-labs`, start Jupyter over an SSH port forward, set `KD_COURSE_MODEL_DIR` if models are pre-downloaded), and lesson 1.6 (what to expect, how long, how to read the results, what to do for each problem line).
- [ ] **Step 4: Run** `make test && make e2e && uv run --group labs pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(course): setup lab and lesson 1.6`

**Known limit:** the notebook cannot be executed in this build environment (no GPU, no model downloads). Its code paths are covered by the CPU tests in Tasks 1 and 2; a run on a real A10G is a manual check for the course owner before release.
