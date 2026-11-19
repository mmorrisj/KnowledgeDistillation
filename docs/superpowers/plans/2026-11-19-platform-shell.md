# Course Platform Shell Implementation Plan (Plan 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working lesson site in the Studio design, served by FastAPI, with one sample lesson that exercises every core component and saves progress in the browser.

**Architecture:** Astro builds MDX lessons to static files in `site/dist`. A FastAPI app serves those files and a small `/api`. Progress goes through a `ProgressStore` interface; this plan ships only the browser-storage implementation, and Plan 2 adds the server-backed one.

**Tech Stack:** Python 3.13, uv, FastAPI, pydantic-settings, pytest, httpx. Node 22, Astro 7, `@astrojs/mdx`, `@astrojs/react`, React, TypeScript, Vitest, Testing Library, Playwright.

**Spec:** `docs/superpowers/specs/2026-11-19-kd-course-platform-design.md`

**Plan series:** (1) this plan; (2) tracking mode: JWT identity, Postgres, instructor page; (3) Module 1 lessons, `kd_course` package and the setup lab. Each produces working software alone.

## Global Constraints

- No containers anywhere (no Dockerfile, no Compose).
- `TRACKING_ENABLED` defaults to `false`. With it off the server never imports or connects to a database.
- A reader of the hosted site needs neither Node nor a database.
- Fonts are bundled with the site (`@fontsource`), never loaded from a third-party host.
- Visual direction is "Studio": Bricolage Grotesque for display, Inter for body, JetBrains Mono for code; deep teal accent with a warm orange highlight; light and dark themes; 14px radius.
- Accessibility: full keyboard use, visible focus, `prefers-reduced-motion` respected, WCAG AA contrast in both themes.
- One `pyproject.toml` at the repo root, managed with uv. Dependency groups: `server`, `dev`. (`labs` arrives in Plan 3.)
- Astro APIs change between major versions. Before using an Astro API, check it against the docs for the installed version.

## Review Focus

1. Browser storage is unavailable or holds corrupt data (private window, blocked site data): lessons still render and progress works for the session. (Task 5)
2. The server starts before the site is built: it exits with a message that names the missing folder and the build command, not a stack trace. (Task 2)
3. A student enters a bad Jupyter address (`javascript:`, no scheme, trailing slashes): only `http`/`https` are accepted and the saved value is normalized. (Task 7)
4. An unknown page returns the styled 404 page; an unknown `/api/...` path returns JSON 404, never HTML. (Task 2)
5. The site is hosted under a path prefix behind a gateway: every internal link and asset honors the prefix. (Tasks 3 and 4)

---

### Task 1: Python project and FastAPI skeleton

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `server/__init__.py`, `server/config.py`, `server/main.py`
- Test: `tests/server/test_config_api.py`

**Interfaces:**
- Produces: `server.config.Settings` (pydantic-settings; fields `tracking_enabled: bool = False`, `site_dist: Path = Path("site/dist")`, `base_path: str = ""`; env names `TRACKING_ENABLED`, `SITE_DIST`, `BASE_PATH`).
- Produces: `server.main.create_app(settings: Settings | None = None) -> FastAPI`.
- Produces: `GET /api/health` → `{"status": "ok"}`; `GET /api/config` → `{"tracking_enabled": false}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_health(client):            # client built from create_app(Settings(site_dist=tmp_dist))
    assert client.get("/api/health").json() == {"status": "ok"}

def test_config_defaults_to_tracking_off(client):
    assert client.get("/api/config").json() == {"tracking_enabled": False}

def test_tracking_flag_read_from_env(monkeypatch):
    monkeypatch.setenv("TRACKING_ENABLED", "true")
    assert Settings().tracking_enabled is True
```

- [ ] **Step 2: Run** `uv run pytest tests/server -v`. Expected: FAIL (module not found).
- [ ] **Step 3: Create `pyproject.toml`** (project name `kd-course`, Python `>=3.12`; group `server`: fastapi, uvicorn[standard], pydantic-settings; group `dev`: pytest, httpx, ruff) **and implement `Settings` and `create_app`.** `.gitignore` covers `.venv`, `node_modules`, `site/dist`, `.astro`, `.superpowers/`.
- [ ] **Step 4: Run** `uv run pytest tests/server -v` and `uv run ruff check .`. Expected: PASS, no lint errors.
- [ ] **Step 5: Commit** `feat(server): FastAPI skeleton with config endpoint`

### Task 2: Serve the built site

**Files:**
- Modify: `server/main.py`
- Create: `server/static.py`, `tests/server/test_static.py`, `tests/server/conftest.py` (fixture `tmp_dist` writes `index.html`, `404.html`, `modules/x/index.html`)

**Interfaces:**
- Consumes: `Settings.site_dist`, `Settings.base_path`.
- Produces: `server.static.mount_site(app: FastAPI, settings: Settings) -> None`; `server.static.SiteNotBuiltError(RuntimeError)`.
- Produces: console entry `kd-course-serve` (`server.main:run`) starting uvicorn on `HOST`/`PORT` (defaults `127.0.0.1`, `8000`).

- [ ] **Step 1: Write the failing tests**

```python
def test_serves_index(client):            assert "<html" in client.get("/").text
def test_serves_nested_page(client):      assert client.get("/modules/x/").status_code == 200
def test_unknown_page_gets_styled_404(client):
    r = client.get("/nope/"); assert r.status_code == 404 and "text/html" in r.headers["content-type"]
def test_unknown_api_path_is_json_404(client):
    r = client.get("/api/nope"); assert r.status_code == 404 and r.json() == {"detail": "Not Found"}
def test_missing_dist_raises_clear_error(tmp_path):
    with pytest.raises(SiteNotBuiltError, match="npm run build"):
        create_app(Settings(site_dist=tmp_path / "missing"))
def test_base_path_prefix(tmp_dist):
    c = TestClient(create_app(Settings(site_dist=tmp_dist, base_path="/kd")))
    assert c.get("/kd/").status_code == 200 and c.get("/kd/api/health").status_code == 200
```

- [ ] **Step 2: Run** `uv run pytest tests/server/test_static.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `mount_site`.** API routes are registered before the static mount; the static handler serves `404.html` with status 404 for misses; `run()` catches `SiteNotBuiltError` and exits with code 1 after printing the message.
- [ ] **Step 4: Run** `uv run pytest -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(server): serve built site with 404 handling and base path`

### Task 3: Astro scaffold, Studio tokens and base layout

**Files:**
- Create: `site/package.json`, `site/astro.config.mjs`, `site/tsconfig.json`, `site/src/styles/tokens.css`, `site/src/styles/base.css`, `site/src/layouts/BaseLayout.astro`, `site/src/pages/index.astro`, `site/src/pages/404.astro`, `site/src/lib/paths.ts`, `site/src/lib/paths.test.ts`, `site/vitest.config.ts`

**Interfaces:**
- Produces: `withBase(path: string): string` in `site/src/lib/paths.ts`, joining `import.meta.env.BASE_URL` and a root-relative path with exactly one slash. All internal links use it.
- Produces: CSS custom properties, light values on `:root`, dark values under `prefers-color-scheme: dark` and `[data-theme="dark"]`: `--bg #f7f8f7/#101615`, `--surface #ffffff/#18201f`, `--ink #14211f/#e3ecea`, `--muted #5b6866/#97a8a5`, `--line #e1e6e4/#2a3533`, `--accent #0e6f66/#4fd1c0`, `--accent-ink #ffffff/#06201d`, `--soft #e3f1ee/#173330`, `--hi #fde4d3/#4a2a17`, `--good #0e6f66/#4fd1c0`, `--bad #c2410c/#fb923c`, `--radius 14px`, `--font-display`, `--font-body`, `--font-mono`.
- Produces: `BaseLayout.astro` props `{ title: string }`, with a default slot and a named `sidebar` slot; two-column above 860px, stacked below.
- Produces: npm scripts `dev`, `build`, `check` (`astro check`), `test` (`vitest run`).
- Astro `base` is read from env `SITE_BASE` (default `/`).

- [ ] **Step 1: Write the failing test** (`paths.test.ts`)

```ts
expect(withBase("/modules/a/", "/")).toBe("/modules/a/");
expect(withBase("/modules/a/", "/kd/")).toBe("/kd/modules/a/");
expect(withBase("modules/a/", "/kd")).toBe("/kd/modules/a/");
```
(`withBase(path, base = import.meta.env.BASE_URL)`.)

- [ ] **Step 2: Scaffold** `site/` with Astro, `@astrojs/mdx`, `@astrojs/react`, `@fontsource-variable/bricolage-grotesque`, `@fontsource-variable/inter`, `@fontsource-variable/jetbrains-mono`, vitest. Run `npm test`. Expected: FAIL (`withBase` missing).
- [ ] **Step 3: Implement** `withBase`, the token and base stylesheets, `BaseLayout`, a home page listing nothing yet, and a 404 page. Styling reference: direction B in the approved mockup (`https://claude.ai/artifact/LwWYtXQxhNWtfrC6N5Qz4e`).
- [ ] **Step 4: Run** `npm test && npm run check && npm run build`, then `SITE_BASE=/kd/ npm run build` and confirm `dist/index.html` references assets under `/kd/`. Expected: all pass.
- [ ] **Step 5: Commit** `feat(site): Astro scaffold with Studio tokens and base layout`

### Task 4: Lesson collection, routing and sidebar

**Files:**
- Create: `site/src/content.config.ts`, `site/src/content/lessons/00-sample/01-components.mdx`, `site/src/pages/modules/[module]/[lesson].astro`, `site/src/layouts/LessonLayout.astro`, `site/src/components/Sidebar.astro`, `site/src/lib/nav.ts`, `site/src/lib/nav.test.ts`, `site/src/content/modules.ts`
- Modify: `site/src/pages/index.astro`

**Interfaces:**
- Produces: lesson front matter schema: `title: string`, `order: number`, `minutes: number`, `objectives: string[]` (1 to 5), `lab?: { title: string, path: string, minutes: number }`. The lesson id is `<module-slug>/<lesson-slug>` taken from the file path.
- Produces: `modules.ts` exporting `MODULES: { slug: string, title: string, status: "available" | "planned" }[]` with the eight modules from the spec plus `00-sample`.
- Produces: `buildNav(lessons: {id: string, title: string, order: number}[], modules: typeof MODULES): NavModule[]` where `NavModule = { slug, title, status, lessons: { id, title, href }[] }`; lessons sorted by `order`; `href` is `/modules/<module>/<lesson>/` before `withBase`.
- Produces: `adjacent(nav: NavModule[], id: string): { prev?: NavLesson, next?: NavLesson }`.
- Produces: `LessonLayout.astro` rendering eyebrow, title, meta chips, objectives box, content slot, prev/next links.

- [ ] **Step 1: Write the failing tests** (`nav.test.ts`)

```ts
it("sorts lessons by order inside their module")      // orders 2,1 → ids in 1,2 order
it("keeps planned modules with an empty lesson list") // status "planned", lessons []
it("builds hrefs as /modules/<module>/<lesson>/")
it("adjacent crosses module boundaries and is undefined at the ends")
it("ignores lessons whose module is not in MODULES")
```

- [ ] **Step 2: Run** `npm test`. Expected: FAIL.
- [ ] **Step 3: Implement** the collection, `buildNav`, `adjacent`, the dynamic route, `Sidebar` (current lesson marked with `aria-current="page"`) and `LessonLayout`. The sample lesson has full front matter and placeholder-free prose explaining what the sample page is for.
- [ ] **Step 4: Run** `npm test && npm run check && npm run build`; confirm `dist/modules/00-sample/01-components/index.html` exists. Add a lesson with `objectives: []` temporarily and confirm the build fails naming the file, then remove it.
- [ ] **Step 5: Commit** `feat(site): lesson collection, routing and sidebar`

### Task 5: Progress store (browser storage)

**Files:**
- Create: `site/src/lib/progress.ts`, `site/src/lib/progress.test.ts`

**Interfaces:**
- Produces:

```ts
export type QuizRecord = { attempts: number; correct: boolean; lastAnswer: string };
export type Progress = { lessons: Record<string, { completedAt: string }>; quizzes: Record<string, QuizRecord> };
export interface ProgressStore {
  load(): Promise<Progress>;
  completeLesson(lessonId: string): Promise<void>;
  recordQuiz(lessonId: string, quizId: string, answer: string, correct: boolean): Promise<void>;
  subscribe(fn: (p: Progress) => void): () => void;
}
export class LocalProgressStore implements ProgressStore   // constructor(storage?: Storage)
export function getProgressStore(): ProgressStore          // singleton; Plan 2 adds the server-backed branch
export function percentComplete(p: Progress, lessonIds: string[]): number  // 0..100, rounded
```
- Storage key `kd-course:progress:v1`. Quiz key `<lessonId>#<quizId>`.

- [ ] **Step 1: Write the failing tests**

```ts
it("starts empty")
it("completeLesson persists and is idempotent (first completedAt is kept)")
it("recordQuiz increments attempts and stores the last answer")
it("once correct, a later wrong answer does not clear correct")
it("notifies subscribers and stops after unsubscribe")
it("falls back to memory when storage.setItem throws")
it("treats corrupt JSON as empty and does not throw")
it("ignores stored data of the wrong shape")
it("percentComplete: 1 of 4 → 25; empty list → 0; ignores unknown ids")
```

- [ ] **Step 2: Run** `npm test`. Expected: FAIL.
- [ ] **Step 3: Implement** `LocalProgressStore`, `getProgressStore`, `percentComplete`.
- [ ] **Step 4: Run** `npm test`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(site): browser-stored progress`

### Task 6: Question components (Quiz, PredictReveal, ReviewCard)

**Files:**
- Create: `site/src/components/Quiz.tsx`, `PredictReveal.tsx`, `ReviewCard.tsx`, matching `*.test.tsx`, `site/src/components/components.css`
- Modify: `site/src/content/lessons/00-sample/01-components.mdx`

**Interfaces:**
- Consumes: `getProgressStore()` from Task 5.
- Produces:

```ts
Quiz:          { lessonId: string; id: string; question: string; options: { text: string; correct?: boolean; feedback: string }[] }
PredictReveal: { id: string; prompt: string; options: string[]; children: ReactNode }   // children shown after a choice
ReviewCard:    { fromModule: string; question: string; children: ReactNode }            // children is the answer
```
- Used in MDX with `client:visible`.

- [ ] **Step 1: Write the failing tests** (Testing Library, jsdom)

```ts
// Quiz
it("shows no feedback before an answer")
it("shows that option's feedback and marks it right or wrong, using text as well as color")
it("calls recordQuiz(lessonId, id, optionText, correct)")
it("restores the last answer from the store on mount")
it("options are buttons reachable by keyboard; feedback is in an aria-live region")
// PredictReveal
it("hides children until an option is chosen, then shows them and keeps the choice marked")
// ReviewCard
it("toggles the answer and updates aria-expanded")
```

- [ ] **Step 2: Run** `npm test`. Expected: FAIL.
- [ ] **Step 3: Implement** the three components, styled from tokens only. Add one of each to the sample lesson.
- [ ] **Step 4: Run** `npm test && npm run check && npm run build`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(site): quiz, predict and review components`

### Task 7: Lesson furniture (WorkedExample, ConceptGrid, LabCard, settings, completion)

**Files:**
- Create: `site/src/components/WorkedExample.astro`, `ConceptGrid.tsx`, `LabCard.tsx`, `CompleteLesson.tsx`, `ProgressBar.tsx`, `site/src/lib/settings.ts`, `site/src/pages/settings.astro`, tests for `settings.ts`, `ConceptGrid`, `LabCard`, `CompleteLesson`
- Modify: `LessonLayout.astro`, `Sidebar.astro`, sample lesson

**Interfaces:**
- Consumes: `ProgressStore`, `percentComplete`, lesson `lab` front matter.
- Produces:

```ts
// settings.ts — storage key "kd-course:settings:v1"
export function normalizeJupyterUrl(input: string): string | null   // http/https only; trims; strips trailing slashes; null if invalid
export function getJupyterUrl(): string | null
export function setJupyterUrl(input: string): boolean               // false if invalid
export function labHref(base: string, notebookPath: string): string // `${base}/lab/tree/${encoded path}`

WorkedExample: { title: string; notes: string[] }   // default slot is a fenced code block; note n pairs with marker n
ConceptGrid:   { rows: string[]; cols: string[]; cells: { row: number; col: number; title: string; summary: string; example: string; bestFor: string; taughtIn: string }[] }
LabCard:       { title: string; path: string; minutes: number }     // shows path and copy button; "Open in Jupyter" link only when a URL is saved, else a link to settings
CompleteLesson:{ lessonId: string }
ProgressBar:   { lessonIds: string[] }
```

- [ ] **Step 1: Write the failing tests**

```ts
// settings
normalizeJupyterUrl("http://localhost:8888///") === "http://localhost:8888"
normalizeJupyterUrl("  https://gpu.example.com/user/a/ ") === "https://gpu.example.com/user/a"
normalizeJupyterUrl("javascript:alert(1)") === null
normalizeJupyterUrl("localhost:8888") === null
normalizeJupyterUrl("") === null
labHref("http://h:8888", "labs/01 intro/a.ipynb") === "http://h:8888/lab/tree/labs/01%20intro/a.ipynb"
// components
it("ConceptGrid shows the first cell's detail at rest and switches on click, with aria-pressed")
it("LabCard shows the notebook path and links to settings when no Jupyter URL is saved")
it("LabCard links to labHref(...) with rel=noopener when a URL is saved")
it("CompleteLesson calls completeLesson and then reads 'Completed'")
it("ProgressBar shows 0% with an empty store and updates when the store notifies")
```

- [ ] **Step 2: Run** `npm test`. Expected: FAIL.
- [ ] **Step 3: Implement.** `LessonLayout` renders `LabCard` from front matter and `CompleteLesson` at the end; `Sidebar` shows `ProgressBar` and a check mark on completed lessons; the settings page has one labelled field with inline validation. The sample lesson's `ConceptGrid` is the 2x2 from the spec (black-box or white-box by full fine-tune or LoRA).
- [ ] **Step 4: Run** `npm test && npm run check && npm run build`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(site): worked example, concept grid, lab card, settings and completion`

### Task 8: One-command run, smoke test, CI and operator guide

**Files:**
- Create: `Makefile`, `tests/e2e/smoke.spec.ts`, `playwright.config.ts`, `.github/workflows/ci.yml`, `docs/operator-guide.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: `kd-course-serve`, npm scripts.
- Produces: `make install` (uv sync with `server` and `dev` groups, `npm ci` in `site/`), `make build`, `make serve`, `make test` (pytest, vitest, astro check), `make e2e`.

- [ ] **Step 1: Write the failing smoke test.** Playwright starts `make serve` on port 8000 (`webServer`), then:

```ts
await page.goto("/modules/00-sample/01-components/");
await page.getByRole("button", { name: /* the sample quiz's correct option */ }).click();
await expect(page.getByText("Correct")).toBeVisible();
await page.getByRole("button", { name: "Mark lesson complete" }).click();
await page.reload();
await expect(page.getByText("Completed")).toBeVisible();          // progress survived reload
await page.goto("/nope/"); await expect(page.getByRole("heading", { name: "Page not found" })).toBeVisible();
// axe: no serious or critical violations on the lesson page in light and dark
```

- [ ] **Step 2: Run** `make e2e`. Expected: FAIL (no Makefile).
- [ ] **Step 3: Implement** the Makefile and Playwright config (`--trace off`, system Chromium via `executablePath: '/opt/pw-browsers/chromium'` only if the default launch fails), CI (jobs: python lint and tests; site test, check, build; e2e), the operator guide (install, build, serve, env vars, serving under a path prefix, a sample systemd unit, and a note that tracking arrives in Plan 2), and a README that points students to the site and operators to the guide.
- [ ] **Step 4: Run** `make test && make e2e`. Expected: PASS.
- [ ] **Step 5: Commit** `chore: Makefile, smoke test, CI and operator guide`
